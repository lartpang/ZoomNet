# -*- coding: utf-8 -*-
# @Time    : 2020
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import torch.nn as nn
from timm.models.layers import to_2tuple


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class ConvGNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        gn_groups=4,
        bias=False,
        act_name="relu",
        inplace=True,
    ):
        """
        执行流程Conv2d => GroupNormalization [=> Activation]

        Args:
            in_planes: 模块输入通道数
            out_planes: 模块输出通道数
            kernel_size: 内部卷积操作的卷积核大小
            stride: 卷积步长
            padding: 卷积padding
            dilation: 卷积的扩张率
            groups: 卷积分组数，需满足pytorch自身要求
            gn_groups: GroupNormalization的分组数，默认为4
            bias: 是否启用卷积的偏置，默认为False
            act_name: 使用的激活函数，默认为relu，设置为None的时候则不使用激活函数
            inplace: 设置激活函数的inplace参数
        """
        super(ConvGNReLU, self).__init__()

        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="gn", module=nn.GroupNorm(num_groups=gn_groups, num_channels=out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name, inplace=inplace))
