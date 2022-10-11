from fractions import Fraction as Fr
from typing import Dict, Sequence, Tuple

import cv2
import numpy as np
from albumentations import DualTransform
from albumentations.augmentations.geometric import functional as F

from torchok.constructor import TRANSFORMS


@TRANSFORMS.register_class
class FitResize(DualTransform):
    """Rescale an image so that it fits in given rectangle, keeping the aspect ratio of the initial image. """
    def __init__(
            self,
            max_height: int = 1024,
            max_width: int = 1024,
            interpolation: int = cv2.INTER_LINEAR,
            always_apply: bool = False,
            p: float = 1,
    ):
        """Initialize FitResize augmentation/
        Accepted targets: image, mask, bboxes, keypoints.
        Accepted image types: uint8, float32.

        Args:
            max_height: maximum size of the image height after the transformation.
            max_width: maximum size of the image width after the transformation.
            interpolation: interpolation method. Default: cv2.INTER_LINEAR.
            p: probability of applying the transform.
        """
        super(FitResize, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.max_height = max_height
        self.max_width = max_width
        self.aspect_ratio = Fr(max_height, max_width)

    def apply(self, img: np.ndarray, max_height: int = 1024, max_width: int = 1024,
              interpolation: int = cv2.INTER_LINEAR, **params) -> np.ndarray:
        height, width = img.shape[:2]
        image_aspect_ratio = Fr(height, width)

        if image_aspect_ratio >= self.aspect_ratio:
            if height > width:
                return F.longest_max_size(img, max_size=max_height, interpolation=interpolation)
            else:
                return F.smallest_max_size(img, max_size=max_height, interpolation=interpolation)
        else:
            if height > width:
                return F.smallest_max_size(img, max_size=max_width, interpolation=interpolation)
            else:
                return F.longest_max_size(img, max_size=max_width, interpolation=interpolation)

    def apply_to_keypoint(self, keypoint: Sequence[float], max_height: int = 1024,
                          max_width: int = 1024, **params) -> Sequence[float]:
        height = params["rows"]
        width = params["cols"]
        image_aspect_ratio = Fr(height, width)

        scale = max_height / height if image_aspect_ratio >= self.aspect_ratio else max_width / width
        return F.keypoint_scale(keypoint, scale, scale)

    def apply_to_bbox(self, bbox: Sequence[float], **params) -> Sequence[float]:
        # Bounding box coordinates are scale invariant
        return bbox

    def get_params(self) -> Dict[str, int]:
        return {"max_height": self.max_height, "max_width": self.max_width}

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "max_height", "max_width", "interpolation"
