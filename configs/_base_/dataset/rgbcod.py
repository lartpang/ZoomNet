datasets = dict(
    train=dict(
        dataset_type="rgb_cod_tr",
        shape=dict(h=256, w=256),
        path=["cod10k_camo_tr"],
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="rgb_cod_te",
        shape=dict(h=256, w=256),
        path=["camo_te", "chameleon", "cpd1k_te", "cod10k_te", "nc4k"],
        interp_cfg=dict(),
    ),
)
