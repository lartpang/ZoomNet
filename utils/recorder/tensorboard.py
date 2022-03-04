# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils.misc import check_mkdir


class TBRecorder(object):
    __slots__ = ["tb"]

    def __init__(self, tb_path):
        check_mkdir(tb_path, delete_if_exists=True)
        self.tb = SummaryWriter(tb_path)

    def record_curve(self, name, data, curr_iter):
        if not isinstance(data, (tuple, list)):
            self.tb.add_scalar(f"data/{name}", data, curr_iter)
        else:
            for idx, data_item in enumerate(data):
                self.tb.add_scalar(f"data/{name}_{idx}", data_item, curr_iter)

    def record_image(self, name, data, curr_iter):
        data_grid = make_grid(data, nrow=data.size(0), padding=5)
        self.tb.add_image(name, data_grid, curr_iter)

    def record_images(self, data_container: dict, curr_iter):
        for name, data in data_container.items():
            data_grid = make_grid(data, nrow=data.size(0), padding=5)
            self.tb.add_image(name, data_grid, curr_iter)

    def record_histogram(self, name, data, curr_iter):
        self.tb.add_histogram(name, data, curr_iter)

    def close_tb(self):
        self.tb.close()
