# -*- coding: utf-8 -*-
# @Time    : 2021/6/1
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import ast
import os.path as osp
import platform
import shutil
import sys
import tempfile
import warnings
from importlib import import_module

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

from utils.misc import check_file_exist, import_modules_from_strings

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
RESERVED_KEYS = ["filename", "text", "pretty_text"]


class _ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(_ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no " f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Configurator:
    """A facility for config and config files.

    Borrow from mmcv.
    """

    @staticmethod
    def _validate_py_syntax(filename):
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError("There are syntax errors in config " f"file {filename}: {e}")

    @staticmethod
    def _file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == "Windows":
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            # Substitute predefined variables
            shutil.copyfile(filename, temp_config_file.name)

            if filename.endswith(".py"):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Configurator._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}
                # delete imported module
                del sys.modules[temp_module_name]
            else:
                raise NotImplementedError
            # close temp file
            temp_config_file.close()

        cfg_text = filename + "\n"
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Configurator._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError("Duplicate key is not allowed among bases")
                base_cfg_dict.update(c)

            base_cfg_dict = Configurator._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False):
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Configurator._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict) and k in b and not v.pop(DELETE_KEY, False):
                allowed_types = (dict, list) if allow_list_keys else dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f"{k}={v} in child config cannot inherit from base "
                        f"because {k} is a dict in the child config but is of "
                        f"type {type(b[k])} in base config. You may set "
                        f"`{DELETE_KEY}=True` to ignore the base config"
                    )
                b[k] = Configurator._merge_a_into_b(v, b[k], allow_list_keys)
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename, import_custom_modules=True):
        cfg_dict, cfg_text = Configurator._file2dict(filename)
        if import_custom_modules and cfg_dict.get("custom_imports", None):
            import_modules_from_strings(**cfg_dict["custom_imports"])
        return Configurator(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def fromstring(cfg_str, file_format):
        """Generate config from config str.

        Args:
            cfg_str (str): Configurator str.
            file_format (str): Configurator file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            obj:`Configurator`: Configurator obj.
        """
        if file_format not in [".py", ".json", ".yaml", ".yml"]:
            raise IOError("Only py/yml/yaml/json type are supported now!")
        if file_format != ".py" and "dict(" in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn('Please check "file_format", the file format may be .py')

        with tempfile.NamedTemporaryFile("w", suffix=file_format) as temp_file:
            temp_file.write(cfg_str)
            temp_file.flush()
            cfg = Configurator.fromfile(temp_file.name)
        return cfg

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        super(Configurator, self).__setattr__("_cfg_dict", _ConfigDict(cfg_dict))
        super(Configurator, self).__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, "r") as f:
                text = f.read()
        else:
            text = ""
        super(Configurator, self).__setattr__("_text", text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f"{k_str}: {v_str}"
            else:
                attr_str = f"{str(k)}={v_str}"
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = "[\n"
                v_str += "\n".join(f"dict({_indent(_format_dict(v_), indent)})," for v_ in v).rstrip(",")
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f"{k_str}: {v_str}"
                else:
                    attr_str = f"{str(k)}={v_str}"
                attr_str = _indent(attr_str, indent) + "]"
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= not str(key_name).isidentifier()
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ""
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += "{"
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = "" if outest_level or is_last else ","
                if isinstance(v, dict):
                    v_str = "\n" + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f"{k_str}: dict({v_str}"
                    else:
                        attr_str = f"{str(k)}=dict({v_str}"
                    attr_str = _indent(attr_str, indent) + ")" + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += "\n".join(s)
            if use_mapping:
                r += "}"
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style="pep8",
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True,
        )
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f"Configurator (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = _ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = _ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super(Configurator, self).__setattr__("_cfg_dict", _cfg_dict)
        super(Configurator, self).__setattr__("_filename", _filename)
        super(Configurator, self).__setattr__("_text", _text)

    def dump(self, file=None):
        cfg_dict = super(Configurator, self).__getattribute__("_cfg_dict").to_dict()
        if self.filename.endswith(".py"):
            if file is None:
                return self.pretty_text
            else:
                with open(file, "w") as f:
                    f.write(self.pretty_text)
        else:
            raise NotImplementedError

    def merge_from_dict(self, options, allow_list_keys=True):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Configurator(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Configurator, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

            # Merge list element
            >>> cfg = Configurator(dict(pipeline=[
            ...     dict(type='LoadImage'), dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Configurator, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type='SelfLoadImage'), dict(type='LoadAnnotations')])

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Default: True.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for subkey in key_list[:-1]:
                d.setdefault(subkey, _ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Configurator, self).__getattribute__("_cfg_dict")
        super(Configurator, self).__setattr__(
            "_cfg_dict", Configurator._merge_a_into_b(option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys)
        )
