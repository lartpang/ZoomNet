import random

import albumentations as A
import numpy as np
from albumentations.core.composition import BaseCompose


class Compose(BaseCompose):
    """Compose transforms and handle all transformations regarding bounding boxes

    Args:
        transforms (list): list of transformations to compose.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
    """

    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1.0):
        super(Compose, self).__init__([t for t in transforms if t is not None], p)

        self.processors = {}
        if bbox_params:
            if isinstance(bbox_params, dict):
                params = A.BboxParams(**bbox_params)
            elif isinstance(bbox_params, A.BboxParams):
                params = bbox_params
            else:
                raise ValueError("unknown format of bbox_params, please use `dict` or `BboxParams`")
            self.processors["bboxes"] = A.BboxProcessor(params, additional_targets)

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                params = A.KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, A.KeypointParams):
                params = keypoint_params
            else:
                raise ValueError("unknown format of keypoint_params, please use `dict` or `KeypointParams`")
            self.processors["keypoints"] = A.KeypointsProcessor(params, additional_targets)

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

    def __call__(self, *args, force_apply=False, **data):
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")
        self._check_args(**data)
        assert isinstance(force_apply, (bool, int)), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors.values():
            p.ensure_data_valid(data)
        transforms = self.transforms if need_to_run else self.transforms.get_always_apply(self.transforms)
        dual_start_end = transforms.start_end if self.processors else None
        check_each_transform = any(
            getattr(item.params, "check_each_transform", False) for item in self.processors.values()
        )

        for idx, t in enumerate(transforms):
            if dual_start_end is not None and idx == dual_start_end[0]:
                for p in self.processors.values():
                    p.preprocess(data)

            data = t(force_apply=force_apply, **data)

            if dual_start_end is not None and idx == dual_start_end[1]:
                for p in self.processors.values():
                    p.postprocess(data)
            elif check_each_transform and isinstance(t, A.DualTransform):
                rows, cols = data["image"].shape[:2]
                for p in self.processors.values():
                    if not getattr(p.params, "check_each_transform", False):
                        continue

                    for data_name in p.data_fields:
                        data[data_name] = p.filter(data[data_name], rows, cols)

        return data

    def _to_dict(self):
        dictionary = super(Compose, self)._to_dict()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict() if bbox_processor else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
            }
        )
        return dictionary

    def get_dict_with_id(self):
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict() if bbox_processor else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
                "params": None,
            }
        )
        return dictionary

    def _check_args(self, **kwargs):
        checked_single = ["image", "mask"]
        checked_multi = ["masks"]
        # ["bboxes", "keypoints"] could be almost any type, no need to check them
        for data_name, data in kwargs.items():
            internal_data_name = self.additional_targets.get(data_name, data_name)
            if internal_data_name in checked_single:
                if not isinstance(data, np.ndarray):
                    raise TypeError("{} must be numpy array type".format(data_name))
            if internal_data_name in checked_multi:
                if data:
                    if not isinstance(data[0], np.ndarray):
                        raise TypeError("{} must be list of numpy arrays".format(data_name))
