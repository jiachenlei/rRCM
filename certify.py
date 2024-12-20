
import os
import datetime
import builtins
from time import time

# for logging
from absl import flags
from absl import app
from absl import logging

import torch
import accelerate

from core import Smooth
import rrcm.utils as utils
from rrcm_tune import reload_forward, load_data, FinetuneModel


class BaseModel(FinetuneModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, xt):
        sigmas = torch.full((xt.shape[0], ), self.sigma, device=xt.device)
        # Find the appropriate time step for certifying at the noise level self.sigma. 
        # E.g., t=[0.5, 1.0, 2.0] for sigma=[0.25, 0.5, 1.0] respectively 
        # The t is 2x bigger than sigma used in ceritification
        # This is because we adopt mean=std=0.5 in data preprocessing pipeline during pre-training.
        idx = self.find_nearest(sigmas) 

        # model prediction
        h = self.forward_features(xt, idx) # input noisy image and time condition to vit encoder
        return self.linear_head(h) # return logits of prediction

    # set the noise level sigma used in certification
    def set_sigma(self, sigma):
        self.sigma = sigma


def gather(lst, accelerator):
    for i in range(len(lst)):
        if isinstance(lst[i], torch.Tensor):
            lst[i] = lst[i].clone().detach().to(accelerator.device)
        else:
            lst[i] = torch.tensor(lst[i], device=accelerator.device)

        lst[i] = accelerator.gather(lst[i])

    return lst


@torch.no_grad()
def train(args):
    # CR hyper-parameters
    n0 = 100
    n = args.n if args.n != -1 else 100000
    assert n in [100000, 10000]

    alpha = 0.001
    max_num = 500 if args.dataset.name == "imagenet" else 500 # maximum number of images to certify
    bs = 200 if args.dataset.name == "imagenet" else 500  # batch size used in certification, not data loading

    # logging
    workdir = "./workdir"
    os.makedirs(os.path.join(workdir, f"{args.name}"), exist_ok=True)

    # initialize accelerator
    accelerator = accelerate.Accelerator(mixed_precision="no", split_batches=False)
    device = accelerator.device

    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=os.path.join(workdir, f'{args.name}.log'))
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    logging.info(args)
    sigmas = torch.tensor(args.sigmas)
    logging.info(f"sigmas for certification:{sigmas}")
    time_sigmas = torch.tensor([sigma*2 for sigma in args.sigmas])
    logging.info(f"time steps used: {time_sigmas}",)

    # prepare model and diffusion class for denoising
    ema_scale_fn = utils.create_ema_and_scales_fn(
        **args.ema_scale
    )
    steps = int(args.step)
    ema, num_scales, tau = ema_scale_fn(steps)

    model, diffusion = utils.create_model(**args.nnet), utils.create_diffusion(**args.diffusion)
    model.forward = reload_forward(model, layernorm=args.train.layernorm)

    class_num = 1000 if args.dataset.name == "imagenet" else 10
    diffusion.set_scale(num_scales)
    base_model  = BaseModel(base_model=model, diffusion=diffusion, class_num=class_num, num_scales=num_scales, 
                    sigma_max=args.diffusion.sigma_max, sigma_min=args.diffusion.sigma_min)
    state_dict = torch.load(args.path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        if "module" in k:
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    base_model.load_state_dict(new_state_dict, strict=True)
    base_model.to(device)
    base_model.eval()

    # certify noise level \in sigmas
    for i, sigma in enumerate(sigmas):
        open(os.path.join(workdir, f"{args.name}/certify_result_{sigma}.txt"), "w").close() # empty content in file
        logfp = open(os.path.join(workdir, f"{args.name}/certify_result_{sigma}.txt"), "a+")
        base_model.set_sigma(time_sigmas[i]) # find appropriate time step for certifing on noise level sigma
        certifier = Smooth(base_model, class_num, sigma)

        test_dataset = load_data(
            name = args.dataset.name,
            data_dir=args.dataset.data_dir,
            batch_size=1,
            image_size=args.dataset.image_size,
            mode="test",
            value_range = "0,1", # No normalization when load data. The augmentation is implemented after adding noise to image, see #103 in core.py
        )

        # record the labels of sampled set of images
        # this is for make suring that the same set of images are certified
        # when running in single process or DDP mode 
        debug_all_y = {}
        for i in range(max_num):
            _, y = test_dataset[i]
            if y not in debug_all_y:
                debug_all_y[y] = 0
            debug_all_y[y] += 1
        logging.info(f"data to be certified: {debug_all_y}")

        # generate data with fixed generator
        # this ensures images certified by each process are not duplicate
        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        sampler = torch.utils.data.SubsetRandomSampler(
            indices=[i for i in range(max_num)],
            generator=g_cpu,
        )
        loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            drop_last=False,
            batch_size=1,
            sampler = sampler,
        )
        # distribute data for each process if DDP enabled
        loader = accelerator.prepare(loader)

        debug_all_y = {} # record labels of the actual certified images in the run
        certified_num = 0 # record number of images that have been certified
        for x, y in loader:

            x = x.to(device)

            before_time = time()
            prediction, radius = certifier.certify(x, n0, n, alpha, batch_size=bs)
            after_time = time()
            correct = int(prediction == y)
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

            y, prediction, radius, correct = gather([y, prediction, radius, correct], accelerator)
            if accelerator.is_main_process:
                for i in range(accelerator.num_processes):
                    if certified_num < max_num:
                        # When run certification in DDP mode and max_num % number_of_process != 0
                        # this would lead to some images are certified repeatedly, slightly inflating the results
                        # As a result, we stop recording results when certified_num == max_number
                        print("{}\t{}\t{:.3}\t{}\t{}".format(
                            y[i].item(), prediction[i].item(), radius[i].item(), correct[i].item(), time_elapsed), file=logfp, flush=True)

                        # record labels of certified images
                        if y[i].item() not in debug_all_y:
                            debug_all_y[y[i].item()] = 0
                        debug_all_y[y[i].item()] += 1

                    certified_num += 1

        logging.info(f"data actually certified: {debug_all_y}")
        logfp.close()


flags.DEFINE_integer("step", -1, help="the step of pre-training, used to determine the number of discretization steps")
flags.DEFINE_integer("n", -1, help="number of smoothing noise")
flags.DEFINE_list("sigmas", [0.25, 0.5, 1.0],  help="noise levels for certification")

def main(argv):
    # other flags is defined in rcm_tune.py
    FLAGS = flags.FLAGS
    config = FLAGS.config
    config.path = FLAGS.path
    config.name = FLAGS.name
    config.step = FLAGS.step
    config.sigmas = FLAGS.sigmas
    config.n = FLAGS.n
    train(config)


if __name__ == "__main__":
    app.run(main)