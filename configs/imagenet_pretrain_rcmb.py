import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    # global variables
    _image_size = 224
    _train_steps = 600000

    config.seed = 1234
    config.name = ""
    config.resume = ""
    config.logdir = ""
    config.path = ""

    config.scale_lr = False
    config.train = d(
        total_training_steps=_train_steps,
        ema_rate = 0.99,
        log_interval=10,
        save_interval=50000,
        gradclip_method="after_warmup",
        gradclip=2.0,
    )

    config.lr_scheduler = d(
        name="warmup-cosine",
        warmup_steps = 15000, # lr scheduler
        total_training_steps=_train_steps,
        min_scale=-1.,
    )

    config.optimizer = d(
        lr = 2.0e-4,
        weight_decay = 0.03,
    )

    config.ema_scale = d(
        # ema
        target_ema_mode="sigmoid",
        start_ema=0.99,
        end_ema = 0.9999,
        ema_decay_steps=_train_steps,
        target_ema_A = 10,
        # scale
        scale_mode="icm",
        start_scales=20,
        end_scales=80,
        total_steps=_train_steps,
    )

    config.diffusion = d(
        sigma_data = .5,
        sigma_max = 80.0,
        sigma_min = 0.002,

        rescale_t=True,
        tau = 0.2,
        collect_across_process = True,
        consistency_target_0ema = True,

    )

    config.nnet = d(
        in_channels=3,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        mlp_time_embed=False,

        image_size=_image_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes = -1,
        use_checkpoint= False,
        p_uncond=0.0,

        hidden_dim=4096,
        output_dim = 256,

        stop_grad_conv1 = True,
        moco_initialization = True,

        drop=0., attn_drop=0., drop_path=0.,

    )

    config.dataset = d(
        name='imagenet',
        image_size=_image_size,
        data_dir='PATH TO TRAIN/',
        batch_size = 4096,
        value_range = "0.5,0.5",
        augmentation_type="strong",
        num_workers = 8,
    )

    return config