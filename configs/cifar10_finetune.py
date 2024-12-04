import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    # global variables
    _image_size = 32
    _train_steps = 600000
    _num_classes = -1
    _class_cond = True if _num_classes >0 else False

    config.seed = 1234
    config.name = ""
    config.resume = ""
    config.logdir = "/mnt/vepfs/home//output"

    config.dist_eval = True
    config.lr_scheduler = d(
        name="warmup-cosine",
        warmup_steps=10, # warm-up epoch
        total_training_steps = 100,
    )
    config.train = d(
        epochs=100,
        layernorm=False,
    )

    config.optimizer = d(
        name = "adamw", 
        param = d(
            lr = 1.5e-4,
            weight_decay = 0.,
            betas=(0.9, 0.999),
        ),
        lr_layer_decay=0.85,
        scale_lr=True,
    )

    config.ema_scale = d(
        # ema
        target_ema_mode="fixed",
        target_ema_A = 10,
        start_ema=0.99,
        end_ema=0.99,
        # scale
        scale_mode="icm",
        start_scales=20,
        end_scales=80,
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

        anchor_timesteps= 81,

    )

    config.nnet = d(
        in_channels=3,
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        mlp_time_embed=False,

        model_type="vit",
        # final_norm = "layernorm",
        image_size=_image_size,
        patch_size=4,
        embed_dim=768,
        hidden_dim=2048,
        output_dim=256,
        depth=12,
        num_heads=12,
        num_classes = _num_classes,
        use_checkpoint= False,
        p_uncond=0.0,
        use_bn = True,
        last_bn = False,
        projection_layers = 3,

        max_scale=81,
        multibnmlp = False,
        final_norm="none",
        pretrain=True,

        drop=0., attn_drop=0., 

        drop_path=0.0,
    )


    config.dataset = d(
        name="cifar10",
        image_size=_image_size,
        data_dir='/mnt/vepfs/home//cifar10',
        batch_size = 1024,
        value_range = "0.5,0.5",

        augmentation_type = "strong",
        mixup = 0.0,
        cutmix = 0.0,
        mixup_prob = 1.0,
        cutmix_minmax = None,
        mixup_switch_prob = 0.5,
        mixup_mode="batch", #  "batch", "pair", or "elem"
        autoaug = "rand-m9-mstd0.5-inc1",
        reprob=0.0,
        label_smoothing = 0.0, 
    )

    return config