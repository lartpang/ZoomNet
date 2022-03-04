datasets = dict(
    train=dict(
        dataset_type="rgbd_sod_tr",
        shape=dict(h=256, w=256),
        path=["njudtrdmra_ori", "nlprtrdmra_ori", "dutrgbdtr"],
        interp_cfg=dict(),
    ),
    test=dict(
        dataset_type="rgbd_sod_te",
        shape=dict(h=256, w=256),
        path=["dutrgbdte", "njudtedmra", "nlprtedmra", "lfsd", "rgbd135", "sip", "ssd", "stereo1000"],
        interp_cfg=dict(),
    ),
)
