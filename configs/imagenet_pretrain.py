import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    # global variables
    _image_size = 224
    _train_steps = 600000
    _num_classes = -1
    _end_scale = 80
    _class_cond = True if _num_classes >0 else False

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
        training_mode = "rcm",
    )
    config.lr_scheduler = d(
        name="warmup-cosine",
        warmup_steps = 10000, # lr scheduler
        total_training_steps=_train_steps,
        min_scale=-1.,
    )

    config.optimizer = d(
        lr = 1.5e-4,
        weight_decay = 0.0,
        # betas=(0.99, 0.999),
    )

    config.ema_scale = d(
        # ema
        target_ema_mode="fixed",
        start_ema=0.99,
        end_ema = 0.9999,
        ema_decay_steps=_train_steps,
        target_ema_A = 10,
        # scale
        scale_mode="icm",
        start_scales=20,
        end_scales=_end_scale,
        total_steps=_train_steps,
    )

    config.diffusion = d(
        weight_schedule="uniform",
        schedule_sampler = "singular_t",
        sigma_data = .5,
        sigma_max = 80.0,
        sigma_min = 0.002,

        rescale_t=True,
        aug_sigma = 0.2,
        tau1 = 0.2,
        tau2 = 0.2,
        collect_across_process = True,

        l2_reg = 0.0,
        attn_reg = False,
        use_dcl = False,
        anchor_timesteps = 81,
        apply_weight = False,
        apply_linear_assignment = False,

        consistency_target_0ema=True,

        temperature_param = d(
            A=0.2, w=0.3, b=0.1
        ),

        sample_param = d(
            p_mean= .0,
            p_std = 2.0,
        ),

        weight_param = d(
            p_mean= .5,
            p_std = 1.5,
            w = 10.,
            b = .8,
        ),

        momentum_queue_size = -1,

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
        num_classes = _num_classes,
        use_checkpoint= False,
        p_uncond=0.0,

        use_bn = True,
        last_bn = False,
        projection_layers = 3,
        hidden_dim=4096,
        output_dim = 256,

        stop_grad_conv1 = True,
        moco_initialization = True,


        pretrain=True,
        final_norm = "none",

        drop=0., attn_drop=0., drop_path=0.,
        adabnmlp = False,
        multibnmlp = False,
        multiclstoken = False,
        max_scale = _end_scale,

    )

    config.dataset = d(
        name='imagenet',
        image_size=_image_size,
        data_dir='',
        class_cond=_class_cond,
        batch_size = 4096,
        value_range = "0.5,0.5",
        augmentation_type="strong",
        num_workers = 8,
    )

    return config