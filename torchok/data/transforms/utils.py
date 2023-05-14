from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import albumentations as alb
from albumentations.core.composition import TransformsSeqType

from torchok.constructor import TRANSFORMS


class MultiView(alb.Compose):
    """Transforms an image into multiple views.

    Args:
        branch_transforms: A sequence of transforms. Every transform creates a new view.
        transforms: A sequence of transforms applied on output of every view.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
    """

    def __init__(
            self,
            branch_transforms: TransformsSeqType,
            transforms: TransformsSeqType,
            bbox_params: Optional[dict] = None,
            keypoint_params: Optional[dict] = None,
            additional_targets: Optional[Dict[str, str]] = None,
            p: float = 1.0,
    ):
        super().__init__(transforms, bbox_params, keypoint_params, additional_targets, p)
        self.branch_transforms = branch_transforms

    def __call__(self, *args, force_apply: bool = False, **data) -> Dict[str, List[Any]]:
        """Transforms an image into multiple views.

        Every transform in self.branch_transforms creates a new view.
        """
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")

        result = defaultdict(list)
        for transform in self.branch_transforms:
            view = super(MultiView, self).__call__(**transform(**data))
            for k, v in view.items():
                result[k].append(v)

        return result


@TRANSFORMS.register_class
class MultiCrop(MultiView):
    """Implements the multi-crop transformations.

    Args:
        crop_sizes: Size of the input image in pixels for each crop category.
        crop_counts: Number of crops for each crop category.
        crop_min_scales: Min scales for each crop category.
        crop_max_scales: Max_scales for each crop category.
        transforms: Transforms which are applied to all crops.
        bbox_params (BboxParams): Parameters for bounding boxes transforms
        keypoint_params (KeypointParams): Parameters for keypoints transforms
        additional_targets (dict): Dict with keys - new target name, values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.

    """

    def __init__(
            self,
            crop_sizes: Tuple[int],
            crop_counts: Tuple[int],
            crop_min_scales: Tuple[float],
            crop_max_scales: Tuple[float],
            transforms: TransformsSeqType,
            bbox_params: Optional[dict] = None,
            keypoint_params: Optional[dict] = None,
            additional_targets: Optional[Dict[str, str]] = None,
            p: float = 1.0,
    ):
        if len(crop_sizes) != len(crop_counts):
            raise ValueError(
                "Length of crop_sizes and crop_counts must be equal but are"
                f" {len(crop_sizes)} and {len(crop_counts)}."
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                "Length of crop_sizes and crop_min_scales must be equal but are"
                f" {len(crop_sizes)} and {len(crop_min_scales)}."
            )
        if len(crop_sizes) != len(crop_max_scales):
            raise ValueError(
                "Length of crop_sizes and crop_max_scales must be equal but are"
                f" {len(crop_sizes)} and {len(crop_max_scales)}."
            )

        crop_transforms = []
        for i, crop_size in enumerate(crop_sizes):
            random_resized_crop = alb.RandomResizedCrop(
                height=crop_size,
                width=crop_size,
                scale=(crop_min_scales[i], crop_max_scales[i])
            )

            crop_transforms.extend([random_resized_crop] * crop_counts[i])
        super().__init__(crop_transforms, transforms, bbox_params, keypoint_params, additional_targets, p)
