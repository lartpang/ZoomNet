train = dict(
    batch_size=10,
    num_workers=4,
    use_amp=True,
    num_epochs=100,
    num_iters=30000,
    epoch_based=True,
    lr=0.0001,
    optimizer=dict(
        mode="adamw",
        set_to_none=True,
        group_mode="r3",  # ['trick', 'r3', 'all', 'finetune'],
        cfg=dict(),
    ),
    grad_acc_step=1,
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
        ),
        mode="poly",
        cfg=dict(
            lr_decay=0.9,
            min_coef=0.001,
        ),
    ),
    save_num_models=1,
    ms=dict(
        enable=False,
        extra_scales=[0.75, 1.25, 1.5],
    ),
    grad_clip=dict(
        enable=False,
        mode="value",  # or 'norm'
        cfg=dict(),
    ),
    ema=dict(
        enable=False,
        cmp_with_origin=True,
        force_cpu=False,
        decay=0.9998,
    ),
)
