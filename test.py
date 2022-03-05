# -*- coding: utf-8 -*-

import argparse
import json
import os
import os.path

import numpy as np
import torch
from tqdm import tqdm

from utils import builder, configurator, io, misc, ops, pipeline, recorder


def parse_config():
    parser = argparse.ArgumentParser("Training and evaluation script")
    parser.add_argument("--config", default="./configs/zoomnet/zoomnet.py", type=str)
    parser.add_argument("--datasets-info", default="./configs/_base_/dataset/dataset_configs.json", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--save-path", type=str)
    parser.add_argument("--minmax-results", action="store_true")
    parser.add_argument("--info", type=str)
    args = parser.parse_args()

    config = configurator.Configurator.fromfile(args.config)
    config.use_ddp = False
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.batch_size is not None:
        config.test.batch_size = args.batch_size
    if args.load_from is not None:
        config.load_from = args.load_from
    if args.info is not None:
        config.experiment_tag = args.info
    if args.save_path is not None:
        if os.path.exists(args.save_path):
            if len(os.listdir(args.save_path)) != 0:
                raise ValueError(f"--save-path is not an empty folder.")
        else:
            print(f"{args.save_path} does not exist, create it.")
            os.makedirs(args.save_path)
    config.save_path = args.save_path
    config.test.to_minmax = args.minmax_results

    with open(args.datasets_info, encoding="utf-8", mode="r") as f:
        datasets_info = json.load(f)

    te_paths = {}
    for te_dataset in config.datasets.test.path:
        if te_dataset not in datasets_info:
            raise KeyError(f"{te_dataset} not in {args.datasets_info}!!!")
        te_paths[te_dataset] = datasets_info[te_dataset]
    config.datasets.test.path = te_paths

    config.proj_root = os.path.dirname(os.path.abspath(__file__))
    config.exp_name = misc.construct_exp_name(model_name=config.model_name, cfg=config)
    return config


def test_once(
    model,
    data_loader,
    save_path,
    tta_setting,
    clip_range=None,
    show_bar=False,
    desc="[TE]",
    to_minmax=False,
):
    model.is_training = False
    cal_total_seg_metrics = recorder.CalTotalMetric()

    pgr_bar = enumerate(data_loader)
    if show_bar:
        pgr_bar = tqdm(pgr_bar, total=len(data_loader), ncols=79, desc=desc)
    for batch_id, batch in pgr_bar:
        batch_images = misc.to_device(batch["data"], device=model.device)
        if tta_setting.enable:
            logits = pipeline.test_aug(
                model=model, data=batch_images, strategy=tta_setting.strategy, reducation=tta_setting.reduction
            )
        else:
            logits = model(data=batch_images)
        probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()

        for i, pred in enumerate(probs):
            mask_path = batch["info"]["mask_path"][i]
            mask_array = io.read_gray_array(mask_path, dtype=np.uint8)
            mask_h, mask_w = mask_array.shape

            # here, sometimes, we can resize the prediciton to the shape of the mask's shape
            pred = ops.imresize(pred, target_h=mask_h, target_w=mask_w, interp="linear")

            if clip_range is not None:
                pred = ops.clip_to_normalize(pred, clip_range=clip_range)

            if to_minmax:
                pred = ops.minmax(pred)

            if save_path:  # 这里的save_path包含了数据集名字
                ops.save_array_as_image(data_array=pred, save_name=os.path.basename(mask_path), save_dir=save_path)

            pred = (pred * 255).astype(np.uint8)
            cal_total_seg_metrics.step(pred, mask_array, mask_path)
    fixed_seg_results = cal_total_seg_metrics.get_results()
    return fixed_seg_results


@torch.no_grad()
def testing(model, cfg):
    pred_save_path = None
    for data_name, data_path, loader in pipeline.get_te_loader(cfg):
        if cfg.save_path:
            pred_save_path = os.path.join(cfg.save_path, data_name)
            print(f"Results will be saved into {pred_save_path}")
        seg_results = test_once(
            model=model,
            save_path=pred_save_path,
            data_loader=loader,
            tta_setting=cfg.test.tta,
            clip_range=cfg.test.clip_range,
            show_bar=cfg.test.get("show_bar", False),
            to_minmax=cfg.test.get("to_minmax", False),
        )
        print(f"Results on the testset({data_name}): {misc.mapping_to_str(data_path)}\n{seg_results}")


def main():
    cfg = parse_config()

    model, model_code = builder.build_obj_from_registry(
        registry_name="MODELS", obj_name=cfg.model_name, return_code=True
    )
    io.load_weight(model=model, load_path=cfg.load_from)

    model.device = "cuda:0"
    model.to(model.device)
    model.eval()

    testing(model=model, cfg=cfg)


if __name__ == "__main__":
    main()
