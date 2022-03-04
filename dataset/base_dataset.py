# -*- coding: utf-8 -*-
# @Time    : 2021/5/16
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import random
from collections import abc

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from utils.ops.tensor_ops import cus_sample


class _BaseSODDataset(Dataset):
    def __init__(self, base_shape: dict, extra_scales: tuple = None, interp_cfg: dict = None):
        """
        :param base_shape:
        :param extra_scales: for multi-scale training
        :param interp_cfg: the config of the interpolation, if it is None, the interpolation will not be done.
        """
        super().__init__()
        self.base_shape = base_shape
        if extra_scales is not None and interp_cfg is None:
            raise ValueError("interp_cfg must be True Value when extra_scales is not None.")
        self.extra_scales = extra_scales
        self._sizes = [(base_shape["h"], base_shape["w"])]
        if extra_scales:
            self._sizes.extend(  # 确保是32的整数倍
                [
                    (
                        s * base_shape["h"] // 32 * 32,
                        s * base_shape["w"] // 32 * 32,
                    )
                    for s in extra_scales
                ]
            )

        if not interp_cfg:  # None or {}
            interp_cfg = {}
            self._combine_func = torch.stack
        else:
            print(f"Using multi-scale training strategy with extra scales: {self._sizes}")
            self._combine_func = torch.cat
        self._interp_cfg = interp_cfg
        self._default_cfg = dict(interpolation="bilinear", align_corners=False)

    def _collate(self, batch, parent_key=None):
        """
        borrow from 'torch.utils.data._utils.collate.default_collate'
        """
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            interp_cfg = self._interp_cfg.get(parent_key, None)
            if interp_cfg is None:
                interp_cfg = self._default_cfg
            else:
                interp_cfg["factors"] = self._default_cfg["factors"]
            batch = [cus_sample(it.unsqueeze(0), mode="size", **interp_cfg) for it in batch]
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return self._combine_func(batch, dim=0, out=out)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, (str, bytes)):
            return batch
        elif isinstance(elem, abc.Mapping):
            return {key: self._collate([d[key] for d in batch], parent_key=key) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(*(self._collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError("each element in list of batch should be of equal size")
            transposed = zip(*batch)
            return [self._collate(samples) for samples in transposed]

        raise TypeError("collate_fn: batch must contain tensors, numbers, dicts or lists; found {}".format(elem_type))

    def collate_fn(self, batch):
        if self._interp_cfg:
            self._default_cfg["factors"] = random.choice(self._sizes)
            return self._collate(batch=batch)
        else:
            return default_collate(batch=batch)
