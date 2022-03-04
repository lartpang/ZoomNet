_base_ = [
    "../_base_/common.py",
    "../_base_/train.py",
    "../_base_/test.py",
]

has_test = True
deterministic = True
use_custom_worker_init = False
model_name = "ZoomNet"

train = dict(
    batch_size=22,
    num_workers=4,
    use_amp=True,
    num_epochs=50,
    epoch_based=True,
    lr=0.05,
    optimizer=dict(
        mode="sgd",
        set_to_none=True,
        group_mode="finetune",
        cfg=dict(
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=False,
        ),
    ),
    sche_usebatch=True,
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="f3",
        cfg=dict(
            lr_decay=0.9,
            min_coef=None,
        ),
    ),
    ms=dict(
        enable=True,
        extra_scales=[i / 352 for i in [224, 256, 288, 320, 352]],
    ),
)

test = dict(
    batch_size=22,
    num_workers=4,
    show_bar=False,
)

datasets = dict(
    train=dict(
        dataset_type="msi_sod_tr",
        shape=dict(h=352, w=352),
        path=["dutstr"],
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="msi_sod_te",
        shape=dict(h=352, w=352),
        path=["pascal-s", "ecssd", "hku-is", "dutste", "dut-omron", "socte"],
        interp_cfg=dict(),
    ),
)
