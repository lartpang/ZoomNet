# -*- coding: utf-8 -*-
# @Time    : 2021/5/25
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops.tensor_ops import cus_sample, upsample_add


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(ConvBNReLU(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = ConvBNReLU(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class DenseTransLayer(nn.Module):
    def __init__(self, in_C, out_C):
        super(DenseTransLayer, self).__init__()
        down_factor = in_C // out_C
        self.fuse_down_mul = ConvBNReLU(in_C, in_C, 3, 1, 1)
        self.res_main = DenseLayer(in_C, in_C, down_factor=down_factor)
        self.fuse_main = ConvBNReLU(in_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        feat = self.fuse_down_mul(rgb + depth)
        return self.fuse_main(self.res_main(feat) + feat)


class DDPM(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, kernel_size=3, down_factor=4):
        """DDPM，利用nn.Unfold实现的动态卷积模块
        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            kernel_size (int): 指定的生成的卷积核的大小
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DDPM, self).__init__()
        self.kernel_size = kernel_size
        self.mid_c = out_C // 4
        self.down_input = nn.Conv2d(in_xC, self.mid_c, 1)
        self.branch_1 = DepthDC3x3_1(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.branch_3 = DepthDC3x3_3(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.branch_5 = DepthDC3x3_5(self.mid_c, in_yC, self.mid_c, down_factor=down_factor)
        self.fuse = ConvBNReLU(4 * self.mid_c, out_C, 3, 1, 1)

    def forward(self, x, y):
        x = self.down_input(x)
        result_1 = self.branch_1(x, y)
        result_3 = self.branch_3(x, y)
        result_5 = self.branch_5(x, y)
        return self.fuse(torch.cat((x, result_1, result_3, result_5), dim=1))


class DepthDC3x3_1(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """DepthDC3x3_1，利用nn.Unfold实现的动态卷积模块
        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DepthDC3x3_1, self).__init__()
        self.kernel_size = 3
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DepthDC3x3_3(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """DepthDC3x3_3，利用nn.Unfold实现的动态卷积模块
        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DepthDC3x3_3, self).__init__()
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.kernel_size = 3
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=3, padding=3, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


class DepthDC3x3_5(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """DepthDC3x3_5，利用nn.Unfold实现的动态卷积模块
        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DepthDC3x3_5, self).__init__()
        self.kernel_size = 3
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            DenseLayer(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=5, padding=5, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape([N, xC, self.kernel_size ** 2, xH, xW])
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)


@MODELS.register()
class HDFNet_Res50(BasicModelClass):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = timm.create_model(model_name="resnet50", in_chans=3, features_only=True, pretrained=True)
        self.depth_encoder = timm.create_model(model_name="resnet50", in_chans=1, features_only=True, pretrained=True)

        self.rgb_trans = nn.ModuleDict(
            dict(
                layer0=nn.Conv2d(64, 64, kernel_size=1),
                layer1=nn.Conv2d(256, 64, kernel_size=1),
                layer2=nn.Conv2d(512, 64, kernel_size=1),
                layer3=nn.Conv2d(1024, 64, kernel_size=1),
                layer4=nn.Conv2d(2048, 64, kernel_size=1),
            )
        )
        self.rgbd_trans = nn.ModuleDict(
            dict(
                layer2=DenseTransLayer(512, 64),
                layer3=DenseTransLayer(1024, 64),
                layer4=DenseTransLayer(2048, 64),
            )
        )

        self.upconv32 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = ConvBNReLU(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1)

        self.selfdc_32 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_16 = DDPM(64, 64, 64, 3, 4)
        self.selfdc_8 = DDPM(64, 64, 64, 3, 4)

        self.classifier = nn.Conv2d(32, 1, 1)

    def body(self, rgb_image, depth_image):
        rgb_en_feats = self.rgb_encoder(rgb_image)
        depth_en_feats = self.depth_encoder(depth_image)

        depth_en_feats[2] = self.rgbd_trans["layer2"](rgb_en_feats[2], depth_en_feats[2])
        depth_en_feats[3] = self.rgbd_trans["layer3"](rgb_en_feats[3], depth_en_feats[3])
        depth_en_feats[4] = self.rgbd_trans["layer4"](rgb_en_feats[4], depth_en_feats[4])

        rgb_en_feats[0] = self.rgb_trans["layer0"](rgb_en_feats[0])
        rgb_en_feats[1] = self.rgb_trans["layer1"](rgb_en_feats[1])
        rgb_en_feats[2] = self.rgb_trans["layer2"](rgb_en_feats[2])
        rgb_en_feats[3] = self.rgb_trans["layer3"](rgb_en_feats[3])
        rgb_en_feats[4] = self.rgb_trans["layer4"](rgb_en_feats[4])

        x = self.upconv32(rgb_en_feats[4])  # 1024
        x = self.upconv16(upsample_add(self.selfdc_32(x, depth_en_feats[4]), rgb_en_feats[3]))  # 1024
        x = self.upconv8(upsample_add(self.selfdc_16(x, depth_en_feats[3]), rgb_en_feats[2]))  # 512
        x = self.upconv4(upsample_add(self.selfdc_8(x, depth_en_feats[2]), rgb_en_feats[1]))  # 256
        x = self.upconv2(upsample_add(x, rgb_en_feats[0]))  # 64
        x = self.upconv1(cus_sample(x, mode="scale", factors=2))  # 32
        x = self.classifier(x)
        return dict(seg=x)

    def train_forward(self, data, **kwargs):
        assert not {"image", "depth", "mask"}.difference(set(data)), set(data)

        output = self.body(rgb_image=data["image"], depth_image=data["depth"])
        loss, loss_str = self.cal_loss(all_preds=output, gts=data["mask"])
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output = self.body(rgb_image=data["image"], depth_image=data["depth"])
        return output["seg"]

    def cal_loss(self, all_preds, gts):
        def cal_hel(pred, target, eps=1e-6):
            def edge_loss(pred, target):
                edge = target - F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
                edge[edge != 0] = 1
                # input, kernel_size, stride=None, padding=0
                numerator = (edge * (pred - target).abs_()).sum([2, 3])
                denominator = edge.sum([2, 3]) + eps
                return numerator / denominator

            def region_loss(pred, target):
                # 该部分损失更强调前景区域内部或者背景区域内部的预测一致性
                numerator_fore = (target - target * pred).sum([2, 3])
                denominator_fore = target.sum([2, 3]) + eps

                numerator_back = ((1 - target) * pred).sum([2, 3])
                denominator_back = (1 - target).sum([2, 3]) + eps
                return numerator_fore / denominator_fore + numerator_back / denominator_back

            edge_loss = edge_loss(pred, target)
            region_loss = region_loss(pred, target)
            return (edge_loss + region_loss).mean()

        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            sod_loss = F.binary_cross_entropy_with_logits(
                input=preds, target=cus_sample(gts, mode="size", factors=preds.shape[2:]), reduction="mean"
            )
            losses.append(sod_loss)
            loss_str.append(f"BCE:{sod_loss.item():.5f}")

            hel_loss = cal_hel(pred=preds.sigmoid(), target=cus_sample(gts, mode="size", factors=preds.shape[2:]))
            losses.append(hel_loss)
            loss_str.append(f"HEL:{hel_loss.item():.5f}")
        return sum(losses), " ".join(loss_str)
