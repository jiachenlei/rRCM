import torch
import torch.nn.functional as F
from rrcm.resample import create_named_schedule_sampler


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


class RepresentationKarrasDenoiser:
    def __init__(
        self,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        rescale_t=True, # rescale time condition or not

        tau = 1.0, 
        collect_across_process = True, # whether collect negative samples across process
        num_timesteps = 20,
        device = "cuda",
        consistency_target_0ema = False,

        **kwargs,
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.rescale_t = rescale_t

        self.tau = tau

        self.collect_across_process = collect_across_process

        self.schedule_sampler = create_named_schedule_sampler() # unique time step sampler
        self.num_timesteps = num_timesteps
        self.previous_timesteps = num_timesteps
        self.device = device

        self.consistency_target_0ema = consistency_target_0ema

        self.sigmas = torch.tensor([(self.sigma_max ** (1 / self.rho) + idx / (self.num_timesteps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            )**self.rho for idx in range(0, self.num_timesteps)]).to(self.device)

    def set_scale(self, scale):
        self.num_timesteps = scale
        self.sigmas = torch.tensor([(self.sigma_max ** (1 / self.rho) + idx / (self.num_timesteps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            )**self.rho for idx in range(0, self.num_timesteps)]).to(self.device)

    def to(self, device):
        self.device = device

    def sample_schedule(self, batch_size):
        return self.schedule_sampler.sample(self.num_timesteps, batch_size, self.device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_sigmas(self, sigmas):
        return sigmas

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        _, _, c_in = self.get_scalings_for_boundary_condition(sigmas)
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44) if self.rescale_t else sigmas
        ret = model(append_dims(c_in, x_t.ndim) * x_t, rescaled_t, **model_kwargs)
        return ret

    def rcm_tune_denoise(self, model, x_t, sigmas, indices, **model_kwargs):
        h = self.denoise(model, x_t, sigmas, indices=indices, **model_kwargs)
        return h

    def contrastive_loss(self, f, f_target, 
                         tau=1, collect_across_process = True, accelerator=None,):

        if collect_across_process and accelerator.num_processes > 1:
            rank = accelerator.process_index
            world_size = accelerator.num_processes
            B, C = f_target.shape
            global_ftarget = accelerator.gather(f_target).reshape(world_size, B, C)
            # rearrange batch for correct positive similarity
            all_f_target = [global_ftarget[rank]]
            for i in range(world_size):
                if i != rank:
                    all_f_target.append(global_ftarget[i])
            f_target = torch.cat(all_f_target, dim=0) # cat along batch dimension


        f = F.normalize(f.flatten(1), dim=1)
        f_target = F.normalize(f_target.flatten(1), dim=1)
    
        logits = torch.einsum("nc,ck->nk", [f, f_target.T])
        l_pos = logits.diagonal(0)
        logsumexp_pos = l_pos / tau
        logsumexp_all = torch.logsumexp(logits.float() / tau, dim=1).to(f.dtype)
        nce = logsumexp_all  - logsumexp_pos

        return nce

    def rRCMloss(
        self,
        model,
        x_start,
        num_scales,
        indices=None,
        model_kwargs=None,
        target_model=None,
        noise=None,
        x_aug = None,
        x_aug2 = None,
        accelerator = None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_start)

        assert target_model is not None, "Must have a target model"
        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)

        @torch.no_grad()
        def target_denoise_fn(x, t):
            return self.denoise(target_model, x, t, **model_kwargs)

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        indices0 = torch.full((x_start.shape[0],), self.num_timesteps-1, device=x_start.device)
        t0 = self.sigma_max ** (1 / self.rho) + indices0 / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t0 = t0**self.rho

        # compute symmetric contrastive loss
        x_t0 = x_aug + noise*append_dims(t0, dims)
        x_t0_target = x_aug2 + noise*append_dims(t0, dims)

        h = denoise_fn(x_t0, t0)
        h_target = target_denoise_fn(x_t0_target, t0)
        h = h[0]
        h_target = h_target[0].detach()
        tau = self.tau
        xt_contrast = self.contrastive_loss(h, h_target, 
                                            tau=tau,
                                            collect_across_process=True,
                                            accelerator=accelerator,
                                            )
        h = denoise_fn(x_t0_target, t0)
        h_target = target_denoise_fn(x_t0, t0)
        h = h[0]
        h_target = h_target[0].detach()
        tau = self.tau
        xt_contrast += self.contrastive_loss(h, h_target, 
                                            tau=tau,
                                            collect_across_process=True,
                                            accelerator=accelerator,
                                            )

        # compute consistency loss
        x_t = x_start + noise*append_dims(t, dims)
        x_t2 = x_start + noise*append_dims(t2, dims)
        h = denoise_fn(x_t, t)[1]
        h_target = denoise_fn(x_t2, t2)[1].detach()
        xt_consistency = self.contrastive_loss(h, h_target, 
                                                tau=tau,
                                                collect_across_process=self.collect_across_process,
                                                accelerator=accelerator,
                                                )

        terms = {}
        terms["loss"] = xt_contrast + xt_consistency 
        terms["xt_metric1"] = xt_contrast.detach().clone()
        terms["xt_metric2"] = xt_consistency.detach().clone()

        return terms