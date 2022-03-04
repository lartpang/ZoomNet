# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np


def minmax(data_array: np.ndarray, up_bound: float = None) -> np.ndarray:
    """
    ::

        data_array = (data_array / up_bound)
        if min_value != max_value:
            data_array = (data_array - min_value) / (max_value - min_value)

    :param data_array:
    :param up_bound: if is not None, data_array will devided by it before the minmax ops.
    :return:
    """
    if up_bound is not None:
        data_array = data_array / up_bound
    max_value = data_array.max()
    min_value = data_array.min()
    if max_value != min_value:
        data_array = (data_array - min_value) / (max_value - min_value)
    return data_array


def clip_to_normalize(data_array: np.ndarray, clip_range: tuple = None) -> np.ndarray:
    clip_range = sorted(clip_range)
    if len(clip_range) == 3:
        clip_min, clip_mid, clip_max = clip_range
        assert 0 <= clip_min < clip_mid < clip_max <= 1, clip_range
        lower_array = data_array[data_array < clip_mid]
        higher_array = data_array[data_array > clip_mid]
        if lower_array.size > 0:
            lower_array = np.clip(lower_array, a_min=clip_min, a_max=1)
            max_lower = lower_array.max()
            lower_array = minmax(lower_array) * max_lower
            data_array[data_array < clip_mid] = lower_array
        if higher_array.size > 0:
            higher_array = np.clip(higher_array, a_min=0, a_max=clip_max)
            min_lower = higher_array.min()
            higher_array = minmax(higher_array) * (1 - min_lower) + min_lower
            data_array[data_array > clip_mid] = higher_array
    elif len(clip_range) == 2:
        clip_min, clip_max = clip_range
        assert 0 <= clip_min < clip_max <= 1, clip_range
        if clip_min != 0 and clip_max != 1:
            data_array = np.clip(data_array, a_min=clip_min, a_max=clip_max)
        data_array = minmax(data_array)
    elif clip_range is None:
        data_array = minmax(data_array)
    else:
        raise NotImplementedError
    return data_array


def clip_normalize_scale(array, clip_min=0, clip_max=250, new_min=0, new_max=255):
    array = np.clip(array, a_min=clip_min, a_max=clip_max)
    array = minmax(array) * (new_max - new_min) + new_min
    return array


def save_array_as_image(data_array: np.ndarray, save_name: str, save_dir: str, to_minmax: bool = False):
    """
    save the ndarray as a image

    Args:
        data_array: np.float32 the max value is less than or equal to 1
        save_name: with special suffix
        save_dir: the dirname of the image path
        to_minmax: minmax the array
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    if data_array.dtype != np.uint8:
        if data_array.max() > 1:
            raise Exception("the range of data_array has smoe errors")
        data_array = (data_array * 255).astype(np.uint8)
    if to_minmax:
        data_array = minmax(data_array, up_bound=255)
        data_array = (data_array * 255).astype(np.uint8)
    cv2.imwrite(save_path, data_array)


def imresize(image_array: np.ndarray, target_h, target_w, interp="linear"):
    _interp_mapping = dict(
        linear=cv2.INTER_LINEAR,
        cubic=cv2.INTER_CUBIC,
        nearst=cv2.INTER_NEAREST,
    )
    assert interp in _interp_mapping, f"Only support interp: {_interp_mapping.keys()}"
    resized_image_array = cv2.resize(image_array, dsize=(target_w, target_h), interpolation=_interp_mapping[interp])
    return resized_image_array
