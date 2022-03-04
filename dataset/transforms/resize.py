import albumentations as A
import cv2


class UniResize(A.DualTransform):
    """UniResize the input to the given height and width.

    Args:
        height (int): desired height of the output.
        width (int): desired width of the output.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(UniResize, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return A.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_mask(self, img, interpolation=cv2.INTER_LINEAR, **params):
        return A.resize(img, height=self.height, width=self.width, interpolation=interpolation)

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        scale_x = self.width / width
        scale_y = self.height / height
        return A.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")


def ms_resize(img, scales, base_h=None, base_w=None, interpolation=cv2.INTER_LINEAR):
    assert isinstance(scales, (list, tuple))
    if base_h is None and base_w is None:
        h = img.shape[0]
        w = img.shape[1]
    else:
        h = base_h
        w = base_w
    return [A.resize(img, height=int(h * s), width=int(w * s), interpolation=interpolation) for s in scales]


def ss_resize(img, scale, base_h=None, base_w=None, interpolation=cv2.INTER_LINEAR):
    if base_h is None and base_w is None:
        h = img.shape[0]
        w = img.shape[1]
    else:
        h = base_h
        w = base_w
    return A.resize(img, height=int(h * scale), width=int(w * scale), interpolation=interpolation)
