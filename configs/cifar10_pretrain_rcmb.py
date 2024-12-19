import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    # global variables
    _image_size = 32
    _train_steps = 300000
    _end_scale = 80


    config.seed = 1234
    config.name = ""
    config.resume = ""
    config.logdir = "/mnt/vepfs/home//output"
    config.path=""

    config.scale_lr = False
    config.train = d(
        total_training_steps=_train_steps,
        ema_rate = 0.9999,
        log_interval=10,
        save_interval=50000,
        gradclip = 2.0,
        gradclip_method = "after_warmup", # range in [after_warmup, icm]
    )
    config.lr_scheduler = d(
        warmup_steps = 10000, # lr scheduler
        total_training_steps=_train_steps,
        name="warmup-cosine", # only warmup, no cosine anneling
        min_scale=-1.,
    )

    config.optimizer = d(
        lr = 0.0001,
        weight_decay = 0.,
        # betas=(0.99, 0.999),
    )
    config.ema_scale = d(
        # ema
        target_ema_mode="sigmoid",
        target_ema_A = 15,
        ema_decay_steps = 300000,
        start_ema=0.99,
        end_ema=0.9999,
        tau=0.2,
        # scale
        scale_mode="icm",
        start_scales=20,
        end_scales=_end_scale,
        total_steps=_train_steps,
    )

    config.diffusion = d(
        sigma_data = .5,
        sigma_max = 80.,
        sigma_min = 0.002,
        rho = 7.,
        rescale_t=True,

        tau = 0.2,
        collect_across_process = True,
        consistency_target_0ema = False,
    )

    config.nnet = d(
        in_channels=3,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        mlp_time_embed=False,

        image_size=_image_size,
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        use_checkpoint= False,
        p_uncond=0.0,
        hidden_dim=2048,
        output_dim = 256,

        stop_grad_conv1 = False,
        drop=0., attn_drop=0., drop_path=0.,
    )

    config.dataset = d(
        name='cifar10',
        image_size=_image_size,
        data_dir='PATH TO TRAINING IMAGES',
        batch_size = 2048,
        value_range = "0.5,0.5",
        augmentation_type="strong",
        num_workers = 8,
    )

    return config