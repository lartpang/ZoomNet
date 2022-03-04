# -*- coding: utf-8 -*-
# @Time    : 2021/5/17
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
from typing import Callable

import torch


def replace_module(
    model, source_base_class, *, target_object=None, target_class=None, reload_params_func: Callable = None
):
    if target_object is None:
        if target_class is None and reload_params_func is None:
            raise ValueError("target_class=None and reload_params_func=None can not happen with target_object=None")
    else:
        if not (target_class is None and reload_params_func is None):
            raise ValueError("target_class!=None and reload_params_func!=None can not happen with target_object!=None")

    for tokens, curr_module in model.named_modules():
        if isinstance(curr_module, source_base_class):
            all_tokens = tokens.split(".")
            parent_tokens = all_tokens[:-1]
            target_token = all_tokens[-1]
            curr_attr = model
            for t in parent_tokens:
                curr_attr = getattr(curr_attr, t)
            if target_class and reload_params_func:
                target_attr = getattr(curr_attr, target_token)
                target_object = reload_params_func(target_attr, target_class)
            setattr(curr_attr, target_token, target_object)


@torch.no_grad()
def load_params_for_new_conv(conv_layer, new_conv_layer, in_dim: int):
    o, i, k_h, k_w = new_conv_layer.weight.shape
    ori_weight = conv_layer.weight
    if in_dim < 3:
        new_weight = ori_weight[:, :in_dim]
    else:
        new_weight = torch.repeat_interleave(ori_weight, repeats=in_dim // i + 1, dim=1)[:, :in_dim]
    new_conv_layer.weight = nn.Parameter(new_weight)
    new_conv_layer.bias = conv_layer.bias
