"""
Train a diffusion model on images.
"""

import random
import blobfile as bf
from absl import logging
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussianNoise(object):

    def __init__(self, coeff=2, sigma_min=0.002, sigma_max=80, sigma_data=1, rho=7, scales=20):
        self.coeff = coeff
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.scales = scales

    def sample_sigmas(self):
        # index = torch.randint(0, min(self.scales // self.coeff + 1, self.scales-1) , (1,))
        index = 4
        sigmas = ( self.sigma_max**(1/self.rho) + ( index/(self.scales-1) )*( self.sigma_min**(1/self.rho) - self.sigma_max**(1/self.rho) ) )**self.rho
        return sigmas

    def __call__(self, x):
        noise = torch.randn_like(x)
        sigmas = self.sample_sigmas()
        x = x + sigmas*noise
        return x


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    if pil_image.size[0] == image_size and pil_image.size[1] == image_size:
        return np.array(pil_image)

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class ImageDataset(Dataset):
    def __init__(
        self,
        name,
        resolution,
        image_paths,
        classes=None,
        augmentation_type = "strong",
        value_range="-1,1" # the range of pixel value after normalization, by default is [-1,1]
    ):
        super().__init__()
        self.name = name
        self.resolution = resolution
        self.local_images = image_paths
        self.local_classes = None

        if value_range == "0,1":
            mean = [0, 0, 0]
            std = [1,1,1]
        elif value_range == "0.5,0.5":
            mean = [0.5, 0.5, 0.5]
            std=[0.5, 0.5, 0.5]
        else:
            mean = [0.49139968, 0.48215827, 0.44653124]
            std=[0.24703233, 0.24348505, 0.26158768]

        normalize = transforms.Normalize(mean = mean, std = std)
        logging.info(f"mean: {mean}, std:{std}")

        if name == "cifar10":
            augmentation = transforms.Compose([
                transforms.RandomResizedCrop(self.resolution, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            
            augmentation1 = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.08, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            augmentation2 = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.08, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                transforms.RandomApply([Solarize()], p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            if augmentation_type == "strong":
                self.augmentation = TwoCropsTransform(augmentation1, augmentation2)
            else:
                self.augmentation = TwoCropsTransform(augmentation, augmentation)


        elif name == "imagenet":
            augmentation1 = transforms.Compose([
                transforms.RandomResizedCrop(self.resolution, scale=(0.08, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1) 
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2), 
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            augmentation2 = transforms.Compose([
                transforms.RandomResizedCrop(self.resolution, scale=(0.08, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1) 
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1), 
                transforms.RandomApply([Solarize()], p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

            self.augmentation = TwoCropsTransform(augmentation1, augmentation2)

        self.totensor = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.simple_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        x_aug, x_aug2 = self.augmentation(pil_image)

        cropped_pil= Image.fromarray(center_crop_arr(pil_image, self.resolution))
        x = self.simple_augmentation(cropped_pil)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return x, x_aug, x_aug2, out_dict


def load_data(
    *,
    name,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    num_workers=8,
    **kwargs,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        if name == "cifar10":
            raw_content = open("labels.txt", "r").readlines()
            labels = {}
            for entry in raw_content:
                filename, label = entry.split(",")
                labels[filename] = int(label)
            classes = [labels[path.split("/")[-1]] for path in all_files]
        elif name == "imagenet": # otherwise imagenet
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
    logging.info(f"total training samples: {len(all_files)}")

    dataset = ImageDataset(
        name,
        image_size,
        all_files,
        classes=classes,
        **kwargs,
    )

    logging.info(f"batch size per process:{batch_size}")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(not deterministic), num_workers=num_workers, drop_last=True,
        pin_memory=True, persistent_workers=True,
    )

    return loader