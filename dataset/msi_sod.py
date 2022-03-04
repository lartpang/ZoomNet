# -*- coding: utf-8 -*-
import random
from typing import Dict, List, Tuple

from PIL import Image
from torchvision.transforms import transforms

from dataset.base_dataset import _BaseSODDataset
from utils.builder import DATASETS
from utils.io.genaral import get_datasets_info_with_keys


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


@DATASETS.register(name="msi_sod_te")
class MSISOD_TestDataset(_BaseSODDataset):
    def __init__(self, root: Tuple[str, dict], shape: Dict[str, int], interp_cfg: Dict = None):
        super().__init__(base_shape=shape, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=[root], extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]

        self.to_tensor = transforms.ToTensor()
        self.to_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        image = Image.open(image_path).convert("RGB")

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        image_1_5 = image.resize((int(base_h * 1.5), int(base_w * 1.5)), resample=Image.BILINEAR)
        image_1_0 = image.resize((base_h, base_w), resample=Image.BILINEAR)
        image_0_5 = image.resize((int(base_h * 0.5), int(base_w * 0.5)), resample=Image.BILINEAR)
        image_1_5 = self.to_normalize(self.to_tensor(image_1_5))
        image_1_0 = self.to_normalize(self.to_tensor(image_1_0))
        image_0_5 = self.to_normalize(self.to_tensor(image_0_5))

        return dict(
            data={
                "image1.5": image_1_5,
                "image1.0": image_1_0,
                "image0.5": image_0_5,
            },
            info=dict(
                mask_path=mask_path,
            ),
        )

    def __len__(self):
        return len(self.total_image_paths)


@DATASETS.register(name="msi_sod_tr")
class MSISOD_TrainDataset(_BaseSODDataset):
    def __init__(
        self, root: List[Tuple[str, dict]], shape: Dict[str, int], extra_scales: List = None, interp_cfg: Dict = None
    ):
        super().__init__(base_shape=shape, extra_scales=extra_scales, interp_cfg=interp_cfg)
        self.datasets = get_datasets_info_with_keys(dataset_infos=root, extra_keys=["mask"])
        self.total_image_paths = self.datasets["image"]
        self.total_mask_paths = self.datasets["mask"]

        self.joint_transform = Compose([RandomHorizontallyFlip(), RandomRotate(10)])
        self.to_tensor = transforms.ToTensor()
        self.image_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
        self.to_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        image_path = self.total_image_paths[index]
        mask_path = self.total_mask_paths[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image, mask = self.joint_transform(image, mask)
        image = self.image_transform(image)

        base_h = self.base_shape["h"]
        base_w = self.base_shape["w"]
        image_1_5 = image.resize((int(base_h * 1.5), int(base_w * 1.5)), resample=Image.BILINEAR)
        image_1_0 = image.resize((base_h, base_w), resample=Image.BILINEAR)
        image_0_5 = image.resize((int(base_h * 0.5), int(base_w * 0.5)), resample=Image.BILINEAR)
        image_1_5 = self.to_normalize(self.to_tensor(image_1_5))
        image_1_0 = self.to_normalize(self.to_tensor(image_1_0))
        image_0_5 = self.to_normalize(self.to_tensor(image_0_5))

        mask_1_0 = mask.resize((base_h, base_w), resample=Image.BILINEAR)
        mask_1_0 = self.to_tensor(mask_1_0)
        mask_1_0 = mask_1_0.ge(0.5).float()  # 二值化

        return dict(
            data={
                "image1.5": image_1_5,
                "image1.0": image_1_0,
                "image0.5": image_0_5,
                "mask": mask_1_0,
            }
        )

    def __len__(self):
        return len(self.total_image_paths)
