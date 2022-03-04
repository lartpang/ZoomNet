datasets = dict(
    train=dict(
        dataset_type="rgb_sod_tr",
        shape=dict(h=256, w=256),
        path=["dutstr"],
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="rgb_sod_te",
        shape=dict(h=256, w=256),
        path=["pascal-s", "ecssd", "hku-is", "dutste", "dut-omron", "socte"],
        interp_cfg=dict(),
    ),
)
