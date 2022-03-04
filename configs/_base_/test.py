test = dict(
    batch_size=8,
    num_workers=2,
    eval_func="default_test",
    clip_range=None,
    tta=dict(  # based on the ttach lib
        enable=False,
        reducation="mean",  # 'mean', 'gmean', 'sum', 'max', 'min', 'tsharpen'
        cfg=dict(
            HorizontalFlip=dict(),
            VerticalFlip=dict(),
            Rotate90=dict(angles=[0, 90, 180, 270]),
            Scale=dict(
                scales=[0.75, 1, 1.5],
                interpolation="bilinear",
                align_corners=False,
            ),
            Add=dict(values=[0, 10, 20]),
            Multiply=dict(factors=[1, 2, 5]),
            FiveCrops=dict(crop_height=224, crop_width=224),
            Resize=dict(
                sizes=[0.75, 1, 1.5],
                original_size=224,
                interpolation="bilinear",
                align_corners=False,
            ),
        ),
    ),
)
