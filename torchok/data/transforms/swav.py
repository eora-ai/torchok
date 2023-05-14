from typing import Tuple, Union, Optional, Dict

import albumentations as alb

from torchok.data.transforms.utils import MultiCrop
from torchok.constructor import TRANSFORMS


@TRANSFORMS.register_class
class SwaVTransform(MultiCrop):
    """Implements the multi-crop transformations for SwaV.

    Args:
        crop_sizes: Size of the input image in pixels for each crop category.
        crop_counts: Number of crops for each crop category.
        crop_min_scales: Min scales for each crop category.
        crop_max_scales: Max_scales for each crop category.
        hf_prob: Probability that horizontal flip is applied.
        vf_prob: Probability that vertical flip is applied.
        rr_prob: Probability that random rotation is applied.
        rr_degrees: Range of degrees to select from for random rotation.
            If rr_degrees is None, images randomly are rotated by 90 degrees zero or more times.
            If rr_degrees is a (min, max) tuple, images rotated by a random angle in [min, max].
            If rr_degrees is a single number, images rotated by a random angle in [-rr_degrees, +rr_degrees].
            All rotations are counter-clockwise.
        cj_prob: Probability that color jitter is applied.
        cj_strength: Strength of the color jitter.
            `cj_bright`, `cj_contrast`, `cj_sat`, and `cj_hue` are multiplied by this value.
        cj_bright: How much to jitter brightness.
        cj_contrast: How much to jitter contrast.
        cj_sat: How much to jitter saturation.
        cj_hue: How much to jitter hue.
        gray_prob: Probability of conversion to grayscale.
        gb_prob: Probability of Gaussian blur.
        sigmas: Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.

    """

    def __init__(
            self,
            crop_sizes: Tuple[int, int] = (224, 96),
            crop_counts: Tuple[int, int] = (2, 6),
            crop_min_scales: Tuple[float, float] = (0.14, 0.05),
            crop_max_scales: Tuple[float, float] = (1.0, 0.14),
            hf_prob: float = 0.5,
            vf_prob: float = 0.0,
            rr_prob: float = 0.0,
            rr_degrees: Union[None, int, Tuple[int, int]] = 0,
            cj_prob: float = 0.8,
            cj_strength: float = 1.0,
            cj_bright: float = 0.8,
            cj_contrast: float = 0.8,
            cj_sat: float = 0.8,
            cj_hue: float = 0.2,
            gray_prob: float = 0.2,
            gb_prob: float = 0.5,
            sigmas: Tuple[float, float] = (0.1, 2),
            bbox_params: Optional[dict] = None,
            keypoint_params: Optional[dict] = None,
            additional_targets: Optional[Dict[str, str]] = None,
            p: float = 1.0,
    ):

        transforms = [
            alb.HorizontalFlip(p=hf_prob),
            alb.VerticalFlip(p=vf_prob),
            alb.RandomRotate90(p=rr_prob) if rr_degrees is None else alb.Rotate(limit=rr_degrees, p=rr_prob),
            alb.ColorJitter(
                brightness=cj_strength * cj_bright,
                contrast=cj_strength * cj_contrast,
                saturation=cj_strength * cj_sat,
                hue=cj_strength * cj_hue,
                p=cj_prob
            ),
            alb.ToGray(p=gray_prob),
            alb.GaussianBlur(sigma_limit=sigmas, p=gb_prob)
        ]

        super().__init__(
            crop_sizes=crop_sizes,
            crop_counts=crop_counts,
            crop_min_scales=crop_min_scales,
            crop_max_scales=crop_max_scales,
            transforms=transforms,
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
            additional_targets=additional_targets,
            p=p,
        )
