import torch
import torch.nn as nn
import numpy as np
import os
import collections
from tqdm import tqdm
from absl import logging
from rcm.vit import RCMViT
from rcm.rcm_denoiser import RepresentationKarrasDenoiser

import abc
from torch import nn
import einops
from absl import logging

class MetricLogger(object):

    def __init__(self, **metrics):

        self.metrics = collections.OrderedDict(metrics)

    def add(self, metric_of_the_step:dict):
        for k, v in metric_of_the_step.items():
            if k not in self.metrics:
                self.metrics[k] = {
                    "value": 0
                }
            self.metrics[k]["value"] = v

    def update(self, metric_of_the_step: dict):
        for k, v in metric_of_the_step.items():
            if k not in self.metrics:
                self.metrics[k] = {
                    "value": 0,
                    "cnt": 0,
                }
            oldval, cnt = self.metrics[k]["value"], self.metrics[k]["cnt"] 
            self.metrics[k]["value"] = oldval*cnt/(cnt+1) + v/(cnt+1)
            self.metrics[k]["cnt"] += 1

    def get(self, key=None):
        if key is None:
            return {k:self.metrics[k]["value"] for k in self.metrics.keys()}
        else:
            return self.metrics[key]["value"]

    def _mean(self, metric_list:list):
        return sum(metric_list)/len(metric_list)

    def clean(self):
        self.metrics = collections.OrderedDict({})

    def __repr__(self) -> str:
        return ",".join(["{}:{:03f}".format(k, self._mean(v)) for k,v in self.metrics.items()])


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    end_ema,
    target_ema_A,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    **kwargs,
):
    def ema_and_scales_fn(step):
        enable_gradclip = False
        gradclip_method = kwargs.get("gradclip_method", "after_warmup")
        def icm_scale(step):
            K = total_steps
            k = step
            K_prime = np.ceil(
                K / np.log2(np.ceil(end_scales/start_scales) + 1)
            )
            scales = min(start_scales*(2**np.ceil(k/K_prime)), end_scales) + 1
            return scales

        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "icm":
            scales = icm_scale(step)
            if gradclip_method == "icm":
                prev_scales = icm_scale(step-1) if step -1 > 0 else scales
                if prev_scales != scales:
                    enable_gradclip = True
            target_ema = start_ema
        elif target_ema_mode == "sigmoid" and scale_mode == "icm":
            ema_decay_steps = kwargs.get("ema_decay_steps", total_steps)
            A = target_ema_A
            ema =  np.sqrt(
                    (step / ema_decay_steps) * ( end_ema ** 2 - start_ema**2 )
                    + start_ema**2
                ) if step < ema_decay_steps else end_ema

            c = 2/(1+np.exp(-A*((ema-start_ema)/(end_ema-start_ema)))) - 1

            target_ema = c*end_ema + (1-c)*start_ema
            scales = icm_scale(step)
            if gradclip_method == "icm":
                prev_scales = icm_scale(step-1) if step -1 > 0 else scales
                if prev_scales != scales:
                    enable_gradclip = True

        elif target_ema_mode == "sigmoid" and scale_mode == "fixed":
            ema_decay_steps = kwargs.get("ema_decay_steps", total_steps)

            A = target_ema_A
            ema =  np.sqrt(
                    (step / ema_decay_steps) * ( end_ema ** 2 - start_ema**2 )
                    + start_ema**2
                ) if step < ema_decay_steps else end_ema

            c = 2/(1+np.exp(-A*((ema-start_ema)/(end_ema-start_ema)))) - 1

            target_ema = c*end_ema + (1-c)*start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        else:
            raise NotImplementedError

        return float(target_ema), int(scales), enable_gradclip

    return ema_and_scales_fn


def rcm_model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        image_size=32,
        patch_size=4,
        embed_dim=768, 
        hidden_dim=4096, 
        output_dim=256, 
        depth=12, 
        num_heads=12,
        use_checkpoint=False,
        p_uncond=0.0,

        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data = 0.5,

        collect_across_process = True,
        tau = 0.1,
        rescale_t=True,
    )
    return res


# reserved for debugging
def rcm_train_defaults():
    return dict(
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=600000,
        start_ema=0.0,
        start_scales=20,
        end_scales=80,
    )


def create_diffusion(**kwargs):
    diffusion = RepresentationKarrasDenoiser(**kwargs)
    return diffusion

def create_model(**kwargs):
    return RCMViT(**kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.detach().mul_(rate).add_(p_src, alpha=1 - rate)


class TrainState(object):
    def __init__(self, step, optimizer, lr_scheduler, nnet=None, nnet_ema=None, target_model=None):
        
        self.is_warmup = True
        self.step = step

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.target_model = target_model

    def target_update(self, rate=0.99):
        if self.target_model is not None:
            ema(self.target_model, self.nnet, rate)

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key in ['optimizer', 'lr_scheduler', 'nnet', 'nnet_ema', 'target_model'] and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path, set_step=-1, remove_patch_embed=False):
        logging.info(f'load from {path}')
        try:
            self.step = torch.load(os.path.join(path, 'step.pth'))
        except:
            self.step = 0

        skip_opt_etc = False
        if set_step != -1:
            self.step = set_step
            logging.info(f"The staring step is set to {set_step}")
            skip_opt_etc = True

        for key, val in self.__dict__.items():
            if key not in ['step', 'is_warmup'] and val is not None:
                try:
                    if key in ["nnet", "nnet_ema", "target_model"]:
                        state_dict = torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu')
                        # if remove_patch_embed:
                        #     state_dict.pop("pos_embed")
                        #     state_dict.pop("patch_embed.proj.weight")
                        missing, unexpected = val.load_state_dict(state_dict, strict=False)
                        if len(missing) != 0:
                            logging.info(f"Missing keys:{missing} when loading ckpt of {key}")
                        elif len(unexpected) != 0:
                            # this is expected when loading imagenet for training on cifar10
                            logging.info(f"Unexpected keys:{missing} when loading ckpt of {key}")
                    elif not skip_opt_etc:
                        val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))
                except Exception as ex: # add try except to resume training from a legacy ckpt, trained using codes from consistency model
                    logging.info(f'error when loading ckpt {key}: {ex}, automatically skipping...')

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            logging.info("training from scratch")
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            if "latest.ckpt" in ckpts:
                step = "latest"
            else:
                steps = map(lambda x: int(x.split(".")[0]), ckpts)
                step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            torch.distributed.broadcast(p, 0)


def customized_lr_scheduler(optimizer, min_scale=-1, name="warmup-cosine", warmup_steps=-1, total_training_steps=100000):
    from torch.optim.lr_scheduler import LambdaLR
    import math

    if name == "warmup-cosine":
        def fn(step):
            if warmup_steps > 0:
                if step <= warmup_steps:
                    return min(step / warmup_steps, 1)
                elif step <= total_training_steps:
                    # return lr_min/lr_base + 0.5*(1-lr_min/lr_base)*(1+math.cos(step*math.pi/total_steps))
                    lr_scale = 0.5*(1+math.cos((step-warmup_steps)*math.pi/(total_training_steps-warmup_steps)))
                    if min_scale != -1:
                        lr_scale = max(lr_scale, min_scale)
                    return lr_scale
                else:
                    return min_scale if min_scale != -1 else 0
            else:
                return 1
    elif name == "warmup":
        def fn(step):
            if warmup_steps > 0 and step <= warmup_steps:
                return min(step / warmup_steps, 1)
            else:
                return 1

    return LambdaLR(optimizer, fn)


def param_groups_lrd(model, lr=1e-4, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, ignore=[]):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.base_model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        # print(n)
        if not p.requires_grad or n in ignore:
            continue
        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr": lr*this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr": lr*this_scale,
                "weight_decay": this_decay,
                "params": [],
            }


        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    # print(param_groups.keys())
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers, name2id={}):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in name2id:
        return name2id[name]
    elif name in ['base_model.cls_token', 'base_model.pos_embed']:
        return 0
    elif name.startswith('base_model.patch_embed'):
        return 0
    elif name.startswith('base_model.blocks'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers


def get_optimizer(name, param_groups, **params):
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }

    legal_param = {
        "adam": ["lr", "weight_decay", "betas"],
        "adamw": ["lr", "weight_decay", "betas"],
        "sgd": ["lr", "weight_decay", "momentum"]
    }

    assert name in optimizer_dict.keys(), f"Unsupport optimizer:{name}"
    opt_fn = optimizer_dict[name]

    opt_param = {}
    for n, p in params.items():
        if n in legal_param[name]:
            opt_param[n] = p

    return opt_fn(param_groups, **opt_param)


def initialize_train_state(args, accelerator):

    logging.info("creating model and diffusion...")
    device = accelerator.device

    model = create_model(**args.nnet).train() 
    ema_model= create_model(**args.nnet)
    target_model = create_model(**args.nnet).train()

    for param in target_model.parameters():
        param.requires_grad_(False) # freeze the parameters of target model

    if accelerator.num_processes > 1:
        logging.info("Distributed training, use synchronized batch norm")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ema_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ema_model)
        target_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(target_model)

    optimizer = torch.optim.AdamW(model.parameters(), **args.optimizer)

    train_state = TrainState(step=0, optimizer=optimizer, lr_scheduler=customized_lr_scheduler(optimizer, **args.lr_scheduler),
                             nnet=model, nnet_ema=ema_model, target_model=target_model,
                             )

    train_state.to(device)
    accelerator.wait_for_everyone()

    logging.info("synchronizing model parameters...")
    if accelerator.num_processes > 1:
        sync_params(model.parameters())
        sync_params(model.buffers())

    train_state.ema_update(0)
    train_state.target_update(0)

    return train_state


def log_loss_dict(diffusion, ts, losses):
    metrics = {}
    for key, values in losses.items():
        if key == "loss": continue
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            metrics[f"{key}_q{quartile}"] =  sub_loss

    return metrics


def _compute_norms(state_dict, grad_scale=1.0):
    grad_norm = 0.0
    param_norm = 0.0
    for k, p in state_dict:
        if not p.requires_grad: continue
        with torch.no_grad():
            param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
            if p.grad is not None:
                grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
    return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})

