# -*- coding: utf-8 -*-
# @Time    : 2021/5/31
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import ttach as tta


def test_aug(model, data, strategy, reducation="mean", output_key=None):
    print(f"We will use Test Time Augmentation with {strategy}!")
    merger = tta.base.Merger(type=reducation, n=len(strategy))
    transforms = tta.Compose([getattr(tta, name)(**args) for name, args in strategy.items()])

    for transformer in transforms:
        # augment image
        aug_data = {name: transformer.augment_image(data_item) for name, data_item in data.items()}
        # pass to model
        model_output = model(data=aug_data)
        # reverse augmentation
        if output_key is not None:
            model_output = model_output[output_key]
        deaug_logits = transformer.deaugment_mask(model_output)
        # save results
        merger.append(deaug_logits)

    logits = merger.result
    # reduce results as you want, e.g mean/max/min
    if output_key is not None:
        logits = {output_key: logits}
    return logits
