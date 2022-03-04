# -*- coding: utf-8 -*-
# @Time    : 2021/6/3
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import torch
import torch.nn as nn
from torchvision.models import vgg

from methods.module.base_model import BasicModelClass
from utils.builder import MODELS
from utils.ops.module_ops import load_params_for_new_conv


def Cus_V16BN_tv():
    net = vgg.vgg16_bn(pretrained=True, progress=True)

    head_convs = list(net.children())[0][:6]
    rgb_head = nn.Sequential(*head_convs[:3])
    shared_head = nn.Sequential(*head_convs[3:-1])
    depth_head = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
    load_params_for_new_conv(conv_layer=rgb_head[0], new_conv_layer=depth_head[0], in_dim=1)

    model = nn.ModuleDict(
        dict(
            rgb_head=rgb_head,
            depth_head=depth_head,
            shared_head=shared_head,
            layer1=nn.Sequential(*list(net.children())[0][6:13]),
            layer2=nn.Sequential(*list(net.children())[0][13:23]),
            layer3=nn.Sequential(*list(net.children())[0][23:33]),
            layer4=nn.Sequential(*list(net.children())[0][33:43]),
        )
    )
    return model


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.add_module(
            name="conv",
            module=nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        self.add_module(name="relu", module=nn.ReLU(True))


class DW(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, mode=""):
        super(DW, self).__init__()
        self.loc_3x3_1 = ConvBNReLU(in_dim, mid_dim, 3, 1, 1)
        self.loc_3x3_2 = ConvBNReLU(in_dim, mid_dim, 3, 1, 1)
        self.glo_3x3 = ConvBNReLU(in_dim, mid_dim, 3, 1, 5, dilation=5)
        self.glo_7x7 = ConvBNReLU(in_dim, mid_dim, 7, 1, 3)

        if mode == "up":
            self.fusion = nn.Sequential(
                nn.ConvTranspose2d(4 * mid_dim, out_dim, 2, 2, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.Sigmoid(),
            )
        elif mode == "down":
            self.fusion = nn.Sequential(
                nn.Conv2d(4 * mid_dim, out_dim, 2, 2, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.Sigmoid(),
            )
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(4 * mid_dim, out_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.Sigmoid(),
            )

    def forward(self, fr, fd):
        fd = torch.cat([self.loc_3x3_1(fd), self.loc_3x3_2(fd), self.glo_3x3(fd), self.glo_7x7(fd)], dim=1)
        r_dw = self.fusion(fd)
        return r_dw * fr


class RW(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RW, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid(),
        )

    def forward(self, fr):
        r_rw = self.conv(fr)
        return r_rw * fr


class CMW_LM(nn.Module):
    def __init__(self, in_dims, out_dims, mid_dim):
        super().__init__()
        # Depth-to-RGB weighting (DW)
        self.dw_l = DW(in_dim=max(in_dims), mid_dim=mid_dim, out_dim=min(out_dims), mode="up")
        self.dw_h = DW(in_dim=min(in_dims), mid_dim=mid_dim, out_dim=max(out_dims), mode="down")

        # RGB-to-RGB weighting (RW)
        self.rw_l = RW(in_dim=min(in_dims), out_dim=min(out_dims))
        self.rw_h = RW(in_dim=max(in_dims), out_dim=max(out_dims))

        # Aggregation of Double Weighting Features
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(max(out_dims), min(out_dims), 2, 2),
            nn.BatchNorm2d(min(out_dims)),
            nn.ReLU(True),
            nn.Conv2d(min(out_dims), min(out_dims), 1),
            nn.BatchNorm2d(min(out_dims)),
            nn.ReLU(True),
        )

    def forward(self, rgb_feats, depth_feats):
        fr_l, fr_h = rgb_feats
        fd_l, fd_h = depth_feats

        f_dw_l = self.dw_l(fr=fr_l, fd=fd_h)
        f_rw_l = self.rw_l(fr=fr_l)
        f_de_l = fr_l + f_dw_l + f_rw_l

        f_dw_h = self.dw_h(fr=fr_h, fd=fd_l)
        f_rw_h = self.rw_h(fr=fr_h)
        f_de_h = fr_h + f_dw_h + f_rw_h

        f_cmw = torch.cat([f_de_l, self.up_conv(f_de_h)], dim=1)
        return f_cmw


class CMW_H(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(CMW_H, self).__init__()
        # Depth-to-RGB weighting (DW)
        self.dw = DW(in_dim=in_dim, mid_dim=mid_dim, out_dim=out_dim)

        # RGB-to-RGB weighting (RW)
        self.rw = RW(in_dim=in_dim, out_dim=out_dim)

    def forward(self, rgb_feats, depth_feats):
        fr = rgb_feats
        fd = depth_feats
        f_dw = self.dw(fr=fr, fd=fd)
        f_rw = self.rw(fr=fr)
        f_cmw = fr + f_dw + f_rw
        return f_cmw


@MODELS.register()
class CMWNet_V16(BasicModelClass):
    def __init__(self):
        super().__init__()
        self.siamese_encoder = Cus_V16BN_tv()
        self.cmw_l = CMW_LM(in_dims=(64, 128), mid_dim=64, out_dims=(64, 128))
        self.cmw_m = CMW_LM(in_dims=(256, 512), mid_dim=256, out_dims=(256, 512))
        self.cmw_h = CMW_H(in_dim=512, mid_dim=256, out_dim=512)

        self.d_12 = nn.Sequential(
            ConvBNReLU(256 + 64 * 2, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
            ConvBNReLU(64, 64, 3, 1, 1),
        )
        self.d_34 = nn.Sequential(
            ConvBNReLU(512 + 256 * 2, 256, 3, 1, 1),
            ConvBNReLU(256, 256, 3, 1, 1),
            ConvBNReLU(256, 256, 3, 1, 1),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(256, 256, 4, 4),
        )
        self.d_5 = nn.Sequential(
            ConvBNReLU(512, 512, 3, 1, 1),
            ConvBNReLU(512, 512, 3, 1, 1),
            ConvBNReLU(512, 512, 3, 1, 1),
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(512, 512, 4, 4),
        )

        self.sal_head_12 = nn.Conv2d(64, 1, 3, 1, 1)
        self.sal_head_34 = nn.Conv2d(256, 1, 3, 1, 1)
        self.sal_head_5 = nn.Conv2d(512, 1, 3, 1, 1)

    def body(self, rgb_image, depth_image):
        # separate head
        fr_0 = self.siamese_encoder["rgb_head"](rgb_image)
        fd_0 = self.siamese_encoder["depth_head"](depth_image)
        # siamese body
        fr_0 = self.siamese_encoder["shared_head"](fr_0)
        fd_0 = self.siamese_encoder["shared_head"](fd_0)
        fr_1 = self.siamese_encoder["layer1"](fr_0)
        fd_1 = self.siamese_encoder["layer1"](fd_0)
        fr_2 = self.siamese_encoder["layer2"](fr_1)
        fd_2 = self.siamese_encoder["layer2"](fd_1)
        fr_3 = self.siamese_encoder["layer3"](fr_2)
        fd_3 = self.siamese_encoder["layer3"](fd_2)
        fr_4 = self.siamese_encoder["layer4"](fr_3)
        fd_4 = self.siamese_encoder["layer4"](fd_3)

        d_12 = self.cmw_l(rgb_feats=[fr_0, fr_1], depth_feats=[fd_0, fd_1])
        d_34 = self.cmw_m(rgb_feats=[fr_2, fr_3], depth_feats=[fd_2, fd_3])
        d_5 = self.cmw_h(rgb_feats=fr_4, depth_feats=fd_4)

        d_5 = self.d_5(d_5)
        d_34 = self.d_34(torch.cat([d_34, d_5], dim=1))
        d_12 = self.d_12(torch.cat([d_12, d_34], dim=1))

        sal_12 = self.sal_head_12(d_12)
        sal_34 = self.sal_head_34(d_34)
        sal_5 = self.sal_head_5(fr_4)
        return dict(sal_12=sal_12, sal_34=sal_34, sal_5=sal_5)

    def train_forward(self, data, **kwargs):
        results = self.body(rgb_image=data["image"], depth_image=data["depth"])
        loss, loss_str = self.cal_loss(all_preds=results, gts=data["mask"])
        return results["sal_12"].sigmoid(), loss, loss_str

    def test_forward(self, data, **kwargs):
        results = self.body(rgb_image=data["image"], depth_image=data["depth"])
        return results["sal_12"].sigmoid()
