# -*- coding: utf-8 -*-
# @Time    : 2021/7/25
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
from copy import deepcopy

import torch
from torch import nn


class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, has_set=False, device=None):
        """Model Exponential Moving Average V2 From timm library.

        Keep a moving average of everything in the model state_dict (parameters and buffers).
        V2 of this module is simpler, it does not match params/buffers based on name but simply
        iterates in order. It works with torchscript (JIT of full model).

        This is intended to allow functionality like
        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

        A smoothed version of the weights is necessary for some training schemes to perform well.
        E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
        RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
        smoothing of weights to match results. Pay attention to the decay constant you are using
        relative to your update count per epoch.

        To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
        disable validation of the EMA weights. Validation will have to be done manually in a separate
        process, or after the training stops converging.

        This class is sensitive where it is initialized in the sequence of model init,
        GPU assignment and distributed training wrappers.

        :param model:
        :param decay:
        :param has_set: If the model has a good state, you can set the item to True, otherwise, the update method only
            work when you has called the set method to initialize self.module with a better state.
        :param device:
        """
        super().__init__()
        print(f"[{model.__class__.__name__}] Model Exponential Moving Average with decay: {decay} & device: {device}")
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.has_set = has_set
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    @torch.no_grad()
    def _update(self, model, update_fn):
        if hasattr(model, "module"):
            model = model.module
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if self.device is not None:
                model_v = model_v.to(device=self.device)
            ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, decay=None):
        """
        Use model to update self.module based on a moving average method.
        """
        if self.has_set:
            if decay is None:
                decay = self.decay
            self._update(model, update_fn=lambda e, m: decay * e + (1.0 - decay) * m)

    def set(self, model):
        """
        Use model to initialize self.module.
        """
        self._update(model, update_fn=lambda e, m: m)
        self.has_set = True
