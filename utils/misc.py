# -*- coding: utf-8 -*-
import copy
import importlib
import os
import random
import shutil
import sys
import warnings
from collections import abc, OrderedDict
from datetime import datetime
from importlib.util import find_spec, module_from_spec

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn


def customized_worker_init_fn(worker_id, base_seed):
    set_seed_for_lib(base_seed + worker_id)


def set_seed_for_lib(seed):
    random.seed(seed)
    np.random.seed(seed)
    # 为了禁止hash随机化，使得实验可复现。
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def initialize_seed_cudnn(seed, deterministic):
    assert isinstance(deterministic, bool) and isinstance(seed, int)
    if seed >= 0:
        set_seed_for_lib(seed)
    if not deterministic:
        print("We will use `torch.backends.cudnn.benchmark`")
    else:
        print("We will not use `torch.backends.cudnn.benchmark`")
    cudnn.enabled = True
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic


def construct_path(output_dir: str, exp_name: str) -> dict:
    pth_log_path = os.path.join(output_dir, exp_name)
    tb_path = os.path.join(pth_log_path, "tb")
    save_path = os.path.join(pth_log_path, "pre")
    pth_path = os.path.join(pth_log_path, "pth")

    final_full_model_path = os.path.join(pth_path, "checkpoint_final.pth")
    final_state_path = os.path.join(pth_path, "state_final.pth")

    tr_log_path = os.path.join(pth_log_path, f"tr_{str(datetime.now())[:10]}.txt")
    te_log_path = os.path.join(pth_log_path, f"te_{str(datetime.now())[:10]}.txt")
    trans_log_path = os.path.join(pth_log_path, f"trans_{str(datetime.now())[:10]}.txt")
    cfg_copy_path = os.path.join(pth_log_path, f"cfg_{str(datetime.now())}.py")
    trainer_copy_path = os.path.join(pth_log_path, f"trainer_{str(datetime.now())}.txt")
    excel_path = os.path.join(pth_log_path, f"results.xlsx")

    path_config = {
        "output_dir": output_dir,
        "pth_log": pth_log_path,
        "tb": tb_path,
        "save": save_path,
        "pth": pth_path,
        "final_full_net": final_full_model_path,
        "final_state_net": final_state_path,
        "tr_log": tr_log_path,
        "te_log": te_log_path,
        "trans_log": trans_log_path,
        "cfg_copy": cfg_copy_path,
        "excel": excel_path,
        "trainer_copy": trainer_copy_path,
    }

    return path_config


def construct_exp_name(model_name: str, cfg: dict):
    # bs_16_lr_0.05_e30_noamp_2gpu_noms_352
    focus_item = OrderedDict(
        {
            "train/batch_size": "bs",
            "train/lr": "lr",
            "train/num_epochs": "e",
            "train/num_iters": "i",
            "datasets/train/shape/h": "h",
            "datasets/train/shape/w": "w",
            "train/optimizer/mode": "opm",
            "train/optimizer/group_mode": "opgm",
            "train/scheduler/mode": "sc",
            "train/scheduler/warmup/num_iters": "wu",
            "train/use_amp": "amp",
            # "train/sam/enable": "sam",
            "train/ema/enable": "ema",
            "train/ms/enable": "ms",
        }
    )
    config = copy.deepcopy(cfg)

    def _format_item(_i):
        if isinstance(_i, bool):
            _i = "" if _i else "false"
        elif isinstance(_i, (int, float)):
            if _i == 0:
                _i = "false"
        elif isinstance(_i, (list, tuple)):
            _i = "" if _i else "false"  # 只是判断是否非空
        elif isinstance(_i, str):
            if "_" in _i:
                _i = _i.replace("_", "").lower()
        elif _i is None:
            _i = "none"
        # else: other types and values will be returned directly
        return _i

    if (epoch_based := config.train.get("epoch_based", None)) is not None and (not epoch_based):
        focus_item.pop("train/num_epochs")
    else:
        # 默认基于epoch
        focus_item.pop("train/num_iters")

    exp_names = [model_name]
    for key, alias in focus_item.items():
        item = get_value_recurse(keys=key.split("/"), info=config)
        formatted_item = _format_item(item)
        if formatted_item == 'false':
            continue
        exp_names.append(f"{alias.upper()}{formatted_item}")

    experiment_tag = config.get("experiment_tag", None)
    if experiment_tag:
        exp_names.append(f"INFO{experiment_tag.lower()}")

    return "_".join(exp_names)


def to_device(data, device):
    """
    :param data:
    :param device:
    :return:
    """
    if isinstance(data, (tuple, list)):
        ctor = tuple if isinstance(data, tuple) else list
        return ctor(to_device(d, device) for d in data)
    elif isinstance(data, dict):
        return {name: to_device(item, device) for name, item in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=True)
    else:
        raise TypeError(f"Unsupported type {type(data)}. Only support Tensor or tuple/list/dict containing Tensors.")


def is_on_gpu(x):
    """
    判定x是否是gpu上的实例，可以检测tensor和module
    :param x: (torch.Tensor, nn.Module)目标对象
    :return: 是否在gpu上
    """
    # https://blog.csdn.net/WYXHAHAHA123/article/details/86596981
    if isinstance(x, torch.Tensor):
        return "cuda" in x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).is_cuda
    else:
        raise NotImplementedError


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def get_device(x):
    """
    返回x的设备信息，可以处理tensor和module
    :param x: (torch.Tensor, nn.Module) 目标对象
    :return: 所在设备
    """
    # https://blog.csdn.net/WYXHAHAHA123/article/details/86596981
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, nn.Module):
        return next(x.parameters()).device
    else:
        raise NotImplementedError


def pre_mkdir(path_config):
    # 提前创建好记录文件，避免自动创建的时候触发文件创建事件
    check_mkdir(path_config["pth_log"])
    make_log(path_config["te_log"], f"=== te_log {datetime.now()} ===")
    make_log(path_config["tr_log"], f"=== tr_log {datetime.now()} ===")

    # 提前创建好存储预测结果和存放模型的文件夹
    check_mkdir(path_config["save"])
    check_mkdir(path_config["pth"])


def check_mkdir(dir_name, delete_if_exists=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        if delete_if_exists:
            print(f"{dir_name} will be re-created!!!")
            shutil.rmtree(dir_name)
            os.makedirs(dir_name)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def make_log(path, context):
    with open(path, "a") as log:
        log.write(f"{context}\n")


def are_the_same(file_path_1, file_path_2, buffer_size=8 * 1024):
    """
    通过逐块比较两个文件的二进制数据是否一致来确定两个文件是否是相同内容

    REF: https://zhuanlan.zhihu.com/p/142453128

    Args:
        file_path_1: 文件路径
        file_path_2: 文件路径
        buffer_size: 读取的数据片段大小，默认值8*1024

    Returns: dict(state=True/False, msg=message)
    """
    st1 = os.stat(file_path_1)
    st2 = os.stat(file_path_2)

    # 比较文件大小
    if st1.st_size != st2.st_size:
        return dict(state=False, msg="文件大小不一致")

    with open(file_path_1, mode="rb") as f1, open(file_path_2, mode="rb") as f2:
        while True:
            b1 = f1.read(buffer_size)  # 读取指定大小的数据进行比较
            b2 = f2.read(buffer_size)
            if b1 != b2:
                msg = (
                    f"存在差异:"
                    f"\n{file_path_1}\n==>\n{b1.decode('utf-8')}\n<=="
                    f"\n{file_path_2}\n==>\n{b2.decode('utf-8')}\n<=="
                )
                return dict(state=False, msg=msg)
            # b1 == b2
            if not b1:
                # b1 == b2 == False (b'')
                return dict(state=True, msg="完全一样")


def all_items_in_string(items, target_str):
    """判断items中是否全部都是属于target_str一部分的项"""
    for i in items:
        if i not in target_str:
            return False
    return True


def any_item_in_string(items, target_str):
    """判断items中是否存在属于target_str一部分的项"""
    for i in items:
        if i in target_str:
            return True
    return False


def slide_win_select(items, win_size=1, win_stride=1, drop_last=False):
    num_items = len(items)
    i = 0
    while i + win_size <= num_items:
        yield items[i: i + win_size]
        i += win_stride

    if not drop_last:
        # 对于最后不满一个win_size的切片，保留
        yield items[i: i + win_size]


def iterate_nested_sequence(nested_sequence):
    """
    当前支持list/tuple/int/float/range()的多层嵌套，注意不要嵌套的太深，小心超出python默认的最大递归深度

    例子
    ::

        for x in iterate_nested_sequence([[1, (2, 3)], range(3, 10), 0]):
            print(x)

        1
        2
        3
        3
        4
        5
        6
        7
        8
        9
        0

    :param nested_sequence: 多层嵌套的序列
    :return: generator
    """
    for item in nested_sequence:
        if isinstance(item, (int, float)):
            yield item
        elif isinstance(item, (list, tuple, range)):
            yield from iterate_nested_sequence(item)
        else:
            raise NotImplementedError


def get_value_recurse(keys: list, info: dict):
    curr_key, sub_keys = keys[0], keys[1:]

    if (sub_info := info.get(curr_key, "NoKey")) == "NoKey":
        raise KeyError(f"{curr_key} must be contained in {info}")

    if sub_keys:
        return get_value_recurse(keys=sub_keys, info=sub_info)
    else:
        return sub_info


def import_module_from_module_names(module_names):
    for name in module_names:
        if any([_existing_module.startswith(name) for _existing_module in sys.modules.keys()]):
            print(f"Module:{name} has been contained in sys.modules.")
            continue

        module_spec = find_spec(name)
        if module_spec is None:
            raise ModuleNotFoundError(f"Module :{name} not found")

        print(f"Module:{name} is being imported!")
        module = module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        print(f"Module:{name} has been imported!")


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = importlib.import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def mapping_to_str(mapping: abc.Mapping, *, prefix: str = "    ", lvl: int = 0, max_lvl: int = 1) -> str:
    """
    Print the structural information of the dict.
    """
    sub_lvl = lvl + 1
    cur_prefix = prefix * lvl
    sub_prefix = prefix * sub_lvl

    if lvl == max_lvl:
        sub_items = str(mapping)
    else:
        sub_items = ["{"]
        for k, v in mapping.items():
            sub_item = sub_prefix + k + ": "
            if isinstance(v, abc.Mapping):
                sub_item += mapping_to_str(v, prefix=prefix, lvl=sub_lvl, max_lvl=max_lvl)
            else:
                sub_item += str(v)
            sub_items.append(sub_item)
        sub_items.append(cur_prefix + "}")
        sub_items = "\n".join(sub_items)
    return sub_items


if __name__ == "__main__":
    a = dict(a0=1, b0=2, c0=3, d0=dict(a1=[1, 2, 3], b1=dict(a2=dict(a3=10, b3=[1, 2, 3])), c1=dict(a2=1)))
    print("max_lvl=0 -->\n", mapping_to_str(a, max_lvl=0))
    print("max_lvl=1 -->\n", mapping_to_str(a, max_lvl=1))
    print("max_lvl=2 -->\n", mapping_to_str(a, max_lvl=2))
    print("max_lvl=3 -->\n", mapping_to_str(a, max_lvl=3))
    print("max_lvl=4 -->\n", mapping_to_str(a, max_lvl=4))
    print("max_lvl=5 -->\n", mapping_to_str(a, max_lvl=5))
