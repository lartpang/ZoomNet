# -*- coding: utf-8 -*-
# @Time    : 2021/5/22
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import inspect

from utils.registry import Registry

DATASETS = Registry(name="dataset")
DATASETS.import_module_from_module_names(["dataset"], verbose=False)
MODELS = Registry(name="model")
MODELS.import_module_from_module_names(["methods"], verbose=False)

_ALL_REGISTRY = dict(
    DATASETS=DATASETS,
    MODELS=MODELS,
)


def build_obj_from_registry(obj_name, registry_name, obj_cfg=None, return_code=False):
    """
    :param obj_name:
    :param registry_name: BACKBONES,DATASETS,MODELS,EVALUATE
    :param obj_cfg: For the function obj, if obj_cfg is None, return the function obj, otherwise the outputs of
    the fucntion with the obj_cfg. For the class obj, always return the instance. If obj_cfg is None, it will be
    initialized by an empty, otherwise by the obj_cfg.
    :param return_code: If True, will return the source code of the function or class obj.
    :return: output [, source_code if return_code is True]
    """
    assert (
        registry_name in _ALL_REGISTRY
    ), f"registry_name: {registry_name} must be contained in {_ALL_REGISTRY.keys()}"
    _registry = _ALL_REGISTRY[registry_name]
    assert obj_name in _registry, f"obj_name: {obj_name} must be contained in the registry:\n{_registry}"
    obj = _registry[obj_name]

    if inspect.isclass(obj):
        if obj_cfg is None:
            obj_cfg = {}
        output = obj(**obj_cfg)
    elif inspect.isfunction(obj):
        if obj_cfg is None:
            output = obj
        else:
            output = obj(**obj_cfg)
    else:
        raise NotImplementedError

    if return_code:
        source_code = inspect.getsource(obj)
        return output, source_code
    else:
        return output
