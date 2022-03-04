# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os

import torch
from torch import nn


def save_params(
    model,
    state_net_path,
    model_ema=None,
    full_net_path=None,
    exp_name=None,
    next_epoch=-1,
    optimizer=None,
    scheduler=None,
    total_epoch=-1,
    save_num_models=1,
    scaler=None,
):
    """
    ::

        if isinstance(model, dict):
            model_state = model
        else:
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        opti_state = (optimizer if isinstance(optimizer, dict) else optimizer.state_dict()) if optimizer else None
        sche_state = (scheduler if isinstance(scheduler, dict) else scheduler.state_dict()) if scheduler else None
        scaler_state = (scaler if isinstance(scaler, dict) else scaler.state_dict()) if scaler else None
    """

    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    if full_net_path:
        if next_epoch > 0 and exp_name:
            opti_state = (optimizer if isinstance(optimizer, dict) else optimizer.state_dict()) if optimizer else None
            sche_state = (scheduler if isinstance(scheduler, dict) else scheduler.state_dict()) if scheduler else None
            scaler_state = (scaler if isinstance(scaler, dict) else scaler.state_dict()) if scaler else None
            ema_model_state = (
                (model_ema if isinstance(model_ema, dict) else model_ema.module.state_dict()) if model_ema else None
            )
            torch.save(
                dict(
                    arch=exp_name,
                    epoch=next_epoch,
                    net_state=model_state,
                    ema_net_state=ema_model_state,
                    opti_state=opti_state,
                    sche_state=sche_state,
                    scaler=scaler_state,
                ),
                full_net_path,
            )
        else:
            raise ValueError("!!!NEED: (next_epoch > 0 and exp_name) is True")

    if total_epoch > 0 and save_num_models > 1 and next_epoch >= total_epoch - save_num_models + 1:
        print(f"Saving params of the epoch: {next_epoch - 1}")
        epoch_state_net_path = state_net_path[:-4] + f"_{next_epoch}.pth"
        torch.save(model_state, epoch_state_net_path)
    torch.save(model_state, state_net_path)


def save_weight(save_path, model):
    print(f"Saving weight '{save_path}'")
    if isinstance(model, dict):
        model_state = model
    else:
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(model_state, save_path)
    print(f"Saved weight '{save_path}' " f"(only contain the net's weight)")


def load_specific_params(load_path, names):
    """
    从保存节点恢复参数

    Args:
        load_path (str): 模型存放路径
        names (list): 需要载入的参数名字 [model, optimizer, scheduler, scaler, start_epoch]
    """
    _name_mapping = dict(
        model="net_state",
        optimizer="opti_state",
        scheduler="sche_state",
        scaler="scaler",
        start_epoch="epoch",
        model_ema="ema_net_state",
    )

    assert os.path.exists(load_path) and os.path.isfile(load_path), load_path
    assert all([n in _name_mapping for n in names])

    print(f"Loading parameters from '{load_path}' for {names}")
    checkpoint = torch.load(load_path, map_location=torch.device("cpu"))

    parmas_dict = {}
    for n in names:
        mapped_name = _name_mapping[n]
        if checkpoint.get(mapped_name, None) is not None:
            parmas_dict[n] = checkpoint[mapped_name]
        else:
            raise KeyError(f"There is not '{mapped_name}' in {load_path}: {list(checkpoint.keys())}")
    return parmas_dict


def load_weight(load_path, model: nn.Module):
    """
    从保存节点恢复模型

    Args:
        load_path (str): 模型存放路径
        model: your model
    """
    assert os.path.exists(load_path), load_path

    print(f"Loading weight '{load_path}'")
    ckpt_dict = torch.load(load_path, map_location="cpu")
    state_dict = model.state_dict()
    ckpt_keys = ckpt_dict.keys()
    state_keys = state_dict.keys()
    print(f"Unique Keys in model: {sorted(set(state_keys).difference(ckpt_keys))}")
    print(f"Unique Keys in ckpt: {sorted(set(ckpt_keys).difference(state_keys))}")
    model.load_state_dict(ckpt_dict, strict=False)
    print(f"Loaded weight '{load_path}' " f"(only contains the net's weight)")
