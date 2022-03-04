# -*- coding: utf-8 -*-
import timm
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops.tensor_ops import cus_sample, upsample_add


def down_2x(x):
    return cus_sample(x, mode="scale", factors=0.5)


def up_2x(x):
    return cus_sample(x, mode="scale", factors=2)


def up_to(x, hw):
    return cus_sample(x, mode="size", factors=hw)


class SIM(nn.Module):
    def __init__(self, h_C, l_C):
        super(SIM, self).__init__()
        self.h2l_0 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(l_C)
        self.bnh_0 = nn.BatchNorm2d(h_C)

        self.h2h_1 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(l_C)
        self.bnh_1 = nn.BatchNorm2d(h_C)

        self.h2h_2 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.bnh_2 = nn.BatchNorm2d(h_C)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        h, w = x.shape[2:]

        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(down_2x(x))))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(down_2x(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(up_to(x_l, (h, w)))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(up_to(x_l, (h, w)))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h + x


class conv_2nV1(nn.Module):
    def __init__(self, in_hc=64, in_lc=256, out_c=64, main=0):
        super(conv_2nV1, self).__init__()
        self.main = main
        mid_c = min(in_hc, in_lc)
        self.relu = nn.ReLU(True)

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        self.bnh_1 = nn.BatchNorm2d(mid_c)

        if self.main == 0:
            # stage 2
            self.h2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnh_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.h2h_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnh_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_hc, out_c, 1)

        elif self.main == 1:
            # stage 2
            self.h2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnl_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.l2l_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnl_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_lc, out_c, 1)

        else:
            raise NotImplementedError

    def forward(self, in_h, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(down_2x(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(up_2x(l))
        h = self.relu(self.bnh_1(h2h + l2h))
        l = self.relu(self.bnl_1(l2l + h2l))

        if self.main == 0:
            # stage 2
            l2h = self.l2h_2(up_2x(l))
            h2h = self.h2h_2(h)
            h_fuse = self.relu(self.bnh_2(h2h + l2h))

            # stage 3
            out = self.relu(self.bnh_3(self.h2h_3(h_fuse)) + self.identity(in_h))
        elif self.main == 1:
            # stage 2
            h2l = self.h2l_2(down_2x(h))
            l2l = self.l2l_2(l)
            l_fuse = self.relu(self.bnl_2(h2l + l2l))

            # stage 3
            out = self.relu(self.bnl_3(self.l2l_3(l_fuse)) + self.identity(in_l))
        else:
            raise NotImplementedError

        return out


class conv_3nV1(nn.Module):
    def __init__(self, in_hc=64, in_mc=256, in_lc=512, out_c=64):
        super(conv_3nV1, self).__init__()
        mid_c = min(in_hc, in_mc, in_lc)
        self.relu = nn.ReLU(True)

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.m2m_0 = nn.Conv2d(in_mc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnh_1 = nn.BatchNorm2d(mid_c)
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        self.bnl_1 = nn.BatchNorm2d(mid_c)

        # stage 2
        self.h2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnm_2 = nn.BatchNorm2d(mid_c)

        # stage 3
        self.m2m_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
        self.bnm_3 = nn.BatchNorm2d(out_c)

        self.identity = nn.Conv2d(in_mc, out_c, 1)

    def forward(self, in_h, in_m, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m = self.relu(self.bnm_0(self.m2m_0(in_m)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        m2h = self.m2h_1(up_2x(m))

        h2m = self.h2m_1(down_2x(h))
        m2m = self.m2m_1(m)
        l2m = self.l2m_1(up_2x(l))

        m2l = self.m2l_1(down_2x(m))
        l2l = self.l2l_1(l)

        h = self.relu(self.bnh_1(h2h + m2h))
        m = self.relu(self.bnm_1(h2m + m2m + l2m))
        l = self.relu(self.bnl_1(m2l + l2l))

        # stage 2
        h2m = self.h2m_2(down_2x(h))
        m2m = self.m2m_2(m)
        l2m = self.l2m_2(up_2x(l))
        m = self.relu(self.bnm_2(h2m + m2m + l2m))

        # stage 3
        out = self.relu(self.bnm_3(self.m2m_3(m)) + self.identity(in_m))
        return out


class AIM(nn.Module):
    def __init__(self, iC_list, oC_list):
        super(AIM, self).__init__()
        ic0, ic1, ic2, ic3, ic4 = iC_list
        oc0, oc1, oc2, oc3, oc4 = oC_list
        self.conv0 = conv_2nV1(in_hc=ic0, in_lc=ic1, out_c=oc0, main=0)
        self.conv1 = conv_3nV1(in_hc=ic0, in_mc=ic1, in_lc=ic2, out_c=oc1)
        self.conv2 = conv_3nV1(in_hc=ic1, in_mc=ic2, in_lc=ic3, out_c=oc2)
        self.conv3 = conv_3nV1(in_hc=ic2, in_mc=ic3, in_lc=ic4, out_c=oc3)
        self.conv4 = conv_2nV1(in_hc=ic3, in_lc=ic4, out_c=oc4, main=1)

    def forward(self, *xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        out_xs = []
        out_xs.append(self.conv0(xs[0], xs[1]))
        out_xs.append(self.conv1(xs[0], xs[1], xs[2]))
        out_xs.append(self.conv2(xs[1], xs[2], xs[3]))
        out_xs.append(self.conv3(xs[2], xs[3], xs[4]))
        out_xs.append(self.conv4(xs[3], xs[4]))

        return out_xs


class BaseMINet(BasicModelClass):
    def train_forward(self, data, **kwargs):
        assert not {"image", "mask"}.difference(set(data)), set(data)

        output = self.body(rgb_image=data["image"])
        loss, loss_str = self.cal_loss(all_preds=output, gts=data["mask"])
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output = self.body(rgb_image=data["image"])
        return output["seg"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor):
        """
        loss, loss_str = self.cal_loss(all_preds=output, gts=data["mask"])
        """

        def cal_cel(pred, target):
            pred = pred.sigmoid()
            intersection = pred * target
            numerator = (pred - intersection).sum() + (target - intersection).sum()
            denominator = pred.sum() + target.sum()
            return numerator / (denominator + 1e-6)

        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
            sod_loss = binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")

            cel_loss = cal_cel(pred=preds, target=resized_gts)
            losses.append(cel_loss)
            loss_str.append(f"{name}_CEL: {cel_loss.item():.5f}")
        return sum(losses), " ".join(loss_str)


@MODELS.register()
class MINet_VGG16(BaseMINet):
    def __init__(self):
        super(MINet_VGG16, self).__init__()
        self.encoder = timm.create_model(
            model_name="vgg16_bn", pretrained=True, in_chans=3, features_only=True, output_stride=32
        )

        self.trans = AIM((64, 128, 256, 512, 512), (32, 64, 64, 64, 64))
        self.sim16 = SIM(64, 32)
        self.sim8 = SIM(64, 32)
        self.sim4 = SIM(64, 32)
        self.sim2 = SIM(64, 32)
        self.sim1 = SIM(32, 16)
        self.upconv16 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = ConvBNReLU(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)

    def body(self, rgb_image):
        en_feats = self.encoder(rgb_image)
        en_feats = self.trans(*en_feats)

        x = self.upconv16(self.sim16(en_feats[-1]))  # 1024
        x = self.upconv8(self.sim8(upsample_add(x, en_feats[-2])))  # 512
        x = self.upconv4(self.sim4(upsample_add(x, en_feats[-3])))  # 256
        x = self.upconv2(self.sim2(upsample_add(x, en_feats[-4])))  # 64
        x = self.upconv1(self.sim1(upsample_add(x, en_feats[-5])))  # 32
        x = self.classifier(x)
        return dict(seg=x)


@MODELS.register()
class MINet_Res50(BaseMINet):
    def __init__(self):
        super(MINet_Res50, self).__init__()
        self.encoder = timm.create_model(
            model_name="resnet50", pretrained=True, in_chans=3, features_only=True, output_stride=32
        )
        self.trans = AIM(iC_list=(64, 256, 512, 1024, 2048), oC_list=(64, 64, 64, 64, 64))
        self.sim32 = SIM(64, 32)
        self.sim16 = SIM(64, 32)
        self.sim8 = SIM(64, 32)
        self.sim4 = SIM(64, 32)
        self.sim2 = SIM(64, 32)

        self.upconv32 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = ConvBNReLU(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)

    def body(self, rgb_image):
        en_feats = self.encoder(rgb_image)
        en_feats = self.trans(*en_feats)

        x = self.upconv32(self.sim32(en_feats[-1]))  # 1024
        x = self.upconv16(self.sim16(upsample_add(x, en_feats[-2])))
        x = self.upconv8(self.sim8(upsample_add(x, en_feats[-3])))  # 512
        x = self.upconv4(self.sim4(upsample_add(x, en_feats[-4])))  # 256
        x = self.upconv2(self.sim2(upsample_add(x, en_feats[-5])))  # 64
        x = self.upconv1(cus_sample(x, mode="scale", factors=2))  # 32
        x = self.classifier(x)
        return dict(seg=x)
