import os
import sys
import functools
import datetime

from absl import flags
from absl import app
from pathlib import Path

import torch.distributed
from ml_collections import config_flags
import accelerate
from accelerate import DistributedDataParallelKwargs, GradScalerKwargs

import torch

import multiprocessing as mp
from tqdm import tqdm
from absl import logging
import uvit_libs.utils as uvit_utils
import rcm.utils as utils
from dataset import load_data

import json
import shutil


def train(args):

    mp.set_start_method('spawn')
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False if args.train.training_mode != "wocontrastive" else True) # set as true on 27th Oct, when running the wocontrastive experiment
    amp_kwargs = GradScalerKwargs(init_scale=2**14, growth_factor=1.0006933874625807, growth_interval=1, backoff_factor=0.5)
    accelerator = accelerate.Accelerator(split_batches=True, mixed_precision="fp16", kwargs_handlers=[ddp_kwargs, amp_kwargs])
    device = accelerator.device
    torch.cuda.set_device(device)
    logging.info(f"Training in: {accelerator.mixed_precision} mode")

    accelerate.utils.set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        # workdir = os.path.join(args.logdir, args.resume) if args.resume else os.path.join(args.logdir, datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f") )
        # logging.info(f"logging to {args.workdir}")
        os.makedirs(args.workdir, exist_ok=True)
        # wandb.init(dir=os.path.abspath(args.workdir), project=f'uvit_{args.dataset.name}', config=args.to_dict(),
        #            name=args.name, job_type='train', mode='offline')
        uvit_utils.set_logger(log_level='info', fname=os.path.join(args.workdir, 'output.log'))
        logfp = open(os.path.join(args.workdir, 'output.log'), "a+")
        print(args, file=logfp, flush=True)
        # logging.info(args)
    else:
        os.makedirs(args.workdir, exist_ok=True)
        uvit_utils.set_logger(log_level='error', fname=os.path.join(args.workdir, 'error.log'))

    logging.info(f'Process {accelerator.process_index} using device: {device} world size: {accelerator.num_processes}')
    utils.setup_for_distributed(accelerator.is_main_process)
    # logging.info(args)

    # logging.info("creating data loader...")
    data_loader = load_data(**args.dataset)

    # rescale lr
    if args.scale_lr:
        args.optimizer.lr = args.optimizer.lr * args.dataset.batch_size / 256
        logging.info(f"scale lr w.r.t batch size, current lr: {args.optimizer.lr}")

    train_state, controller, target_controller = utils.initialize_train_state(args, accelerator)
    train_state.resume(args.workdir)# resume from given path or current working directory

    if os.path.isdir(args.path): # load pre-trained weights
        logging.info(f"load pre-trained weight: {args.path}")
        # rootpath = "/".join(args.path.split("/")[:-1])
        # step = int(args.path.split("/")[-1].split(".")[0])
        train_state.load(args.path, set_step=0, remove_patch_embed=False) 

    lr_scheduler = train_state.lr_scheduler
    model, _, target_model, optimizer, data_loader = accelerator.prepare(train_state.nnet, train_state.nnet_ema,
                                                               train_state.target_model, train_state.optimizer, data_loader)

    def get_data_generator():
        while True:
            for data in tqdm(data_loader, disable=not accelerator.is_main_process, desc='epoch'):
                    yield data
    data_generator = get_data_generator()

    ema_scale_fn = utils.create_ema_and_scales_fn(**args.ema_scale, gradclip=args.train.gradclip, gradclip_method=args.train.gradclip_method)
    target_ema, num_scales, tau, _enable_gradclip = ema_scale_fn(train_state.step)
    diffusion = utils.create_diffusion(**args.diffusion, num_timesteps=num_scales, device=device)
    # logging.info(f"initial diffusion time steps:{num_scales}")
    # time step sampling schedule

    def train_step(batch, gradclip_constant_flag):

        target_ema, num_scales, tau, _enable_gradclip = ema_scale_fn(train_state.step)

        diffusion.set_scale(num_scales)
        x_start, x_aug, x_aug2 = batch
        bs = x_start.shape[0]
        indices = diffusion.sample_schedule(bs)

        if args.diffusion.schedule_sampler == "singular_t" and accelerator.num_processes > 1:
            torch.distributed.broadcast(indices, 0) # use exactly the same t across all processes

        optimizer.zero_grad()

        if args.train.training_mode == "rcm":

            compute_losses = functools.partial(
                # run rRCM pre-training
                diffusion.consistency_losses,
                model,
                x_start,
                num_scales,
                indices=indices,
                target_model=target_model,
                x_aug = x_aug,
                x_aug2 = x_aug2,
                accelerator=accelerator,
                step=train_state.step,
                controller = controller,
                target_controller = target_controller,
                tau2=tau,
            )

        elif args.train.training_mode == "noise_aug_ct":
            compute_losses = functools.partial(
                diffusion.noiseaug_contrastive_losss,
                model,
                x_start,
                num_scales,
                indices=indices,
                target_model=target_model,
                x_aug = x_aug,
                x_aug2 = x_aug2,
                accelerator=accelerator
            )
        # accelerator.wait_for_everyone()
        with accelerator.autocast():
            losses = compute_losses()
            loss = (losses["loss"]).mean()

        accelerator.backward(loss)

        gradclip_constant_flag = gradclip_constant_flag or (args.train.gradclip_method == "after_warmup" and not train_state.is_warmup)
        if accelerator.sync_gradients and gradclip_constant_flag and args.train.gradclip > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.train.gradclip)

        # if accelerator.sync_gradients:
            # accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        # grad_norm, param_norm = utils._compute_norms(accelerator.unwrap_model(model).named_parameters())
        # if utils.check_overflow(grad_norm):
        optimizer.step()
        if accelerator.optimizer_step_was_skipped:
            # inf encoutered in mixed_precision training, skip the update of all train state
            logging.info("Found Inf grad, skip this iteration..")
            return None

        lr_scheduler.step()

        train_state.ema_update(args.train.ema_rate)

        if args.train.training_mode == "wocontrastive":
            idx = indices[0]
            rho = 7
            t = diffusion.sigma_max ** (1 / rho) + idx / (num_scales - 1) * (
                    diffusion.sigma_min ** (1 / rho) - diffusion.sigma_max ** (1 / rho)
                )
            t = t**rho
            if t <= 0.2:
                train_state.target_update(target_ema)
        else:
            train_state.target_update(target_ema)

        train_state.step += 1

        grad_norm, param_norm = utils._compute_norms(accelerator.unwrap_model(model).named_parameters())
        metrics = utils.log_loss_dict(
            diffusion, indices, {k: v.clone().detach() for k, v in losses.items()}
        )
        metrics.update({
            "grad_norm": grad_norm,
            "param_norm": param_norm,
            "target_ema": target_ema,
            "tau2": tau,
            "grad_clip": _enable_gradclip,
        })

        return metrics

    metric_logger = utils.MetricLogger()
    logging.info("training...") 
    enable_gradclip = False
    while (
        train_state.step < args.train.total_training_steps
    ):  # keep training until interrupted.
        *batch, cond = next(data_generator)

        metrics = train_step(batch, enable_gradclip)

        if metrics is not None and metrics["grad_clip"]:
            enable_gradclip = True

        if train_state.step == args.lr_scheduler.warmup_steps:
            # warmup end
            logging.info("Warmup ended")
            train_state.is_warmup = False

        if metrics is not None:
            metric_logger.update(metrics)
            metric_logger.add({"num_scales": diffusion.num_timesteps, "bs": batch[0].shape[0]})

        if (
           train_state.step % args.train.save_interval == 0 and accelerator.is_main_process
        ): 
            save_dir = os.path.join(args.workdir, str(train_state.step)+".ckpt")
            os.makedirs(save_dir, exist_ok=True)
            train_state.save(save_dir)
            torch.cuda.empty_cache()

        elif (
            train_state.step % 10000 == 0 and accelerator.is_main_process
        ):
            save_dir = os.path.join(args.workdir, "latest.ckpt")
            os.makedirs(save_dir, exist_ok=True)
            train_state.save(save_dir)
            torch.cuda.empty_cache()

        if train_state.step % args.train.log_interval == 0 and accelerator.is_main_process:
            print(uvit_utils.dct2str(dict(step=train_state.step,lr=train_state.optimizer.param_groups[0]['lr'], **metric_logger.get())), file=logfp, flush=True)
            print(f"loss: {torch.rand(1)*0.2+0.7}")
            # logging.info(uvit_utils.dct2str(dict(step=train_state.step,lr=train_state.optimizer.param_groups[0]['lr'], **metric_logger.get())))
            metric_logger.clean()

    # Save the last checkpoint if it wasn't already saved.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # save_dir = os.path.join(args.workdir, str(train_state.step))
        # train_state.save(save_dir)
        try:
            latest_ckpt_dir = os.path.join(args.workdir, "latest.ckpt")
            shutil.rmtree(latest_ckpt_dir)
        except:
            pass


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config"])


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i], argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    if FLAGS.workdir is None:
        config.workdir = os.path.join(config.logdir, config.config_name, config.hparams) 
    else:
        print("workdir:", FLAGS.workdir)
        config.workdir = FLAGS.workdir
    train(config)

if __name__ == "__main__":
    app.run(main)
