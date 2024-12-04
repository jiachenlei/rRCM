import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from datasets import get_normalize_layer
from torch.nn.functional import interpolate

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["vit", "pretrained-vit", "custom_vit"]

def get_architecture(arch: str, dataset: str, normalize_stats="0.5") -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "custom_vit":
        from vit import ViT
        model = ViT(
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            num_classes = 1000,
        )
    elif arch == "vit":
        from timm.models import vit_base_patch16_224
        model = vit_base_patch16_224(pretrained=False, num_classes=1000)
    elif arch == "pretrained-vit":
        from pytorch_pretrained_vit import ViT
        model = ViT('B_16', pretrained=True, num_classes=1000)
        # model source: https://github.com/lukemelas/PyTorch-Pretrained-ViT
        # pre-trained on ImageNet-21k via supervised learning

    normalize_layer = get_normalize_layer(dataset, normalize_stats=normalize_stats)
    return torch.nn.Sequential(normalize_layer, model)
