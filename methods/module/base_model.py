# -*- coding: utf-8 -*-
# @Time    : 2021/4/14
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import abc

import torch.nn as nn


class BasicModelClass(nn.Module):
    def __init__(self):
        super(BasicModelClass, self).__init__()
        self.is_training = True

    def forward(self, *args, **kwargs):
        if not self.is_training or not self.training:
            results = self.test_forward(*args, **kwargs)
        else:
            results = self.train_forward(*args, **kwargs)
        return results

    @abc.abstractmethod
    def train_forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test_forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def cal_loss(self, *args, **kwargs):
        """
        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            sod_loss = binary_cross_entropy_with_logits(
                input=preds, target=cus_sample(gts, mode="size", factors=preds.shape[2:]), reduction="mean"
            )
            losses.append(sod_loss)
            loss_str.append(f"{name}:{sod_loss.item():.5f}")
        loss = sum(losses)
        loss_str = " ".join(loss_str)
        return loss, loss_str
        """
        pass
