# -*- coding: utf-8 -*-
# @Time    : 2021/5/17
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import contextlib
import json
import os
from collections import defaultdict


def read_data_from_json(json_path: str) -> dict:
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_data_from_txt(path: str) -> list:
    """
    读取文件中各行数据，存放到列表中
    """
    lines = []
    with open(path, encoding="utf-8", mode="r") as f:
        line = f.readline().strip()
        while line:
            lines.append(line)
            line = f.readline().strip()
    return lines


def get_name_list_from_dir(path: str) -> list:
    """直接从文件夹中读取所有文件不包含扩展名的名字"""
    return [os.path.splitext(x)[0] for x in os.listdir(path)]


def get_datasets_info_with_keys(dataset_infos: list, extra_keys: list) -> dict:
    """
    从给定的包含数据信息字典的列表中，依据给定的extra_kers和固定获取的key='image'来获取相应的路径
    Args:
        dataset_infos: 数据集字典
        extra_keys: 除了'image'之外的需要获取的信息名字

    Returns:
        包含指定信息的绝对路径列表
    """

    # total_keys = tuple(set(extra_keys + ["image"]))
    # e.g. ('image', 'mask')
    def _get_intersection(list_a: list, list_b: list, to_sort: bool = True):
        """返回两个列表的交集，并可以随之排序"""
        intersection_list = list(set(list_a).intersection(set(list_b)))
        if to_sort:
            return sorted(intersection_list)
        return intersection_list

    def _get_info(dataset_info: dict, extra_keys: list, path_collection: defaultdict):
        """
        配合get_datasets_info_with_keys使用，针对特定的数据集的信息进行路径获取

        Args:
            dataset_info: 数据集信息字典
            extra_keys: 除了'image'之外的需要获取的信息名字
            path_collection: 存放收集到的路径信息
        """
        total_keys = tuple(set(extra_keys + ["image"]))
        # e.g. ('image', 'mask')

        dataset_root = dataset_info.get("root", ".")

        infos = {}
        for k in total_keys:
            assert k in dataset_info, f"{k} is not in {dataset_info}"
            infos[k] = dict(dir=os.path.join(dataset_root, dataset_info[k]["path"]), ext=dataset_info[k]["suffix"])

        if (index_file_path := dataset_info.get("index_file", None)) is not None:
            image_names = get_data_from_txt(index_file_path)
        else:
            image_names = get_name_list_from_dir(infos["image"]["dir"])

        if "mask" in total_keys:
            mask_names = get_name_list_from_dir(infos["mask"]["dir"])
            image_names = _get_intersection(image_names, mask_names)

        for i, name in enumerate(image_names):
            for k in total_keys:
                path_collection[k].append(os.path.join(infos[k]["dir"], name + infos[k]["ext"]))

    path_collection = defaultdict(list)
    for dataset_name, dataset_info in dataset_infos:
        prev_num = len(path_collection["image"])
        _get_info(dataset_info=dataset_info, extra_keys=extra_keys, path_collection=path_collection)
        curr_num = len(path_collection["image"])
        print(f"Loading data from {dataset_name}: {dataset_info['root']} ({curr_num - prev_num})")
    return path_collection


@contextlib.contextmanager
def open_w_msg(file_path, mode, encoding=None):
    """
    提供了打开关闭的显式提示
    """
    print(f"打开文件{file_path}")
    f = open(file_path, encoding=encoding, mode=mode)
    yield f
    print(f"关闭文件{file_path}")
    f.close()
