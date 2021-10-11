import random
from math import atan2, pi

import cv2
import numpy as np
from albumentations import VerticalFlip, RandomResizedCrop, Compose, normalize_bbox, denormalize_bbox
from albumentations.augmentations import functional as F
from albumentations.augmentations.crops import functional as crop_f
from albumentations.core.transforms_interface import DualTransform

from .utils import convert_to_square, rotate_and_crop_rectangle_safe, \
    rotate_and_crop_keypoints_on_rectangle_safe, keypoints_flip

__all__ = ["RandomCropNearInterestArea", "AlignCropNearInterestArea", "CombineImagesAndCrop",
           "RelativeRandomCrop", "ConditionalTranspose", "RelativePadIfNeeded", "ProportionalRandomResizedCrop",
           "FlipAndConcatTransform"]


class RandomCropNearInterestArea(DualTransform):
    """Crop area with mask if mask is non-empty with random shift by x,y coordinates.

    Args:
        max_part_shift (float): float value in (0.0, 1.0) range. Max relative size of shift in crop. Default 0.3
        min_part_shift (float): float value in (0.0, 1.0) range. Min relative size of shift in crop. Default 0.3
        min_crop_size (tuple): tuple of two values: minimum height and width of crop. Default (0, 0)
        ignore_labels (list of int): values to ignore in mask, `0` values are always ignored
            (e.g. if background value is 5 set `ignore_values=[5]` to ignore)
        p (float): probability of applying the transform. Default: 0.5.

    Params:
        mask: binary mask where 1's label object of interest

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(self, max_part_shift=0.3, min_part_shift=0, min_crop_size=(0, 0),
                 ignore_labels=None, always_apply=False, p=0.5):
        super(RandomCropNearInterestArea, self).__init__(always_apply, p)
        self.max_part_shift = max_part_shift
        self.min_part_shift = min_part_shift
        if isinstance(ignore_labels, int):
            ignore_labels = {ignore_labels}
        elif isinstance(ignore_labels, (list, tuple)):
            ignore_labels = set(ignore_labels)
        self.ignore_labels = {0} if ignore_labels is None else ignore_labels.union({0})
        self.min_crop_size = min_crop_size

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return crop_f.clamping_crop(img, x_min, y_min, x_max, y_max)

    def get_params_dependent_on_targets(self, params):
        mask = params['mask']
        mask = np.where(np.isin(mask, self.ignore_labels), 0, mask)
        img_h, img_w = mask.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            c = random.choice(contours)
            x, y, w, h = cv2.boundingRect(c)
            bbox = x, y, x + w, y + h
            if random.random() > 0.5:
                bbox = convert_to_square(np.array([bbox]))[0]

            h_max_shift = int((bbox[3] - bbox[1]) * self.max_part_shift)
            w_max_shift = int((bbox[2] - bbox[0]) * self.max_part_shift)

            h_min_shift = int((bbox[3] - bbox[1]) * self.min_part_shift)
            w_min_shift = int((bbox[2] - bbox[0]) * self.min_part_shift)

            x_min = bbox[0] - random.randint(w_min_shift, w_max_shift)
            x_max = bbox[2] + random.randint(w_min_shift, w_max_shift)

            y_min = bbox[1] - random.randint(h_min_shift, h_max_shift)
            y_max = bbox[3] + random.randint(h_min_shift, h_max_shift)

            if y_max - y_min < self.min_crop_size[0]:
                y_add = (self.min_crop_size[0] - (y_max - y_min)) // 2
                y_max = min(y_max + y_add, img_h)
                y_min = max(y_min - y_add, 0)
            if x_max - x_min < self.min_crop_size[1]:
                x_add = (self.min_crop_size[1] - (x_max - x_min)) // 2
                x_max = min(x_max + x_add, img_w)
                x_min = max(x_min - x_add, 0)

        else:
            h, w = mask.shape[:2]
            x_min, y_min, x_max, y_max = 0, 0, w, h
        return {'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'rows': h,
                'cols': w
                }

    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return crop_f.bbox_crop(
            bbox, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, rows=params["rows"], cols=params["cols"]
        )

    def apply_to_keypoint(self, keypoint, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return crop_f.crop_keypoint_by_coords(keypoint, crop_coords=(x_min, y_min, x_max, y_max))

    @property
    def targets_as_params(self):
        return ['mask']

    def get_transform_init_args_names(self):
        return 'max_part_shift', 'min_part_shift', 'ignore_labels', 'min_crop_size'


class AlignCropNearInterestArea(DualTransform):
    """
    Rotate image along the mask Crop bbox from image with random shift by x,y coordinates and

    Args:
        max_pad (float): float value in (0.0, 1.0) range. Default 0.3
        min_pad (float): float value in (0.0, 1.0) range. Value must less or equal to the max_pad.
            Default 0.0.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Params:
        interest_mask: binary mask where 1's label object of interest

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, classes_of_interest, max_pad=0.3, min_pad=0, rotate_limit=0,
                 interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        assert max_pad >= min_pad, '`max_pad` must greater or equal to the `min_pad`'
        self.classes_of_interest = classes_of_interest
        self.max_pad = max_pad
        self.min_pad = min_pad
        self.rotate_limit = rotate_limit
        self.interpolation = interpolation

    def apply(self, img, angle=0, box=None, pad=None, **params):
        result = rotate_and_crop_rectangle_safe(img, angle, box, pad, **params)
        return result

    def apply_to_keypoints(self, keypoints, angle=0, box=None, pad=None, **params):
        target_keypoints = keypoints[:, :2]
        meta_inf = keypoints[:, 2:]
        new_keypoints = rotate_and_crop_keypoints_on_rectangle_safe(target_keypoints, angle, box, pad, **params)
        return np.hstack([new_keypoints, meta_inf])

    def get_params_dependent_on_targets(self, params):
        mask = params['mask']
        mask = np.isin(mask, self.classes_of_interest).astype('uint8')
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        h, w = mask.shape

        if len(contours) != 0:
            c = random.choice(contours)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype('int')
            vx, vy, x, y = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = atan2(vy, vx) / pi * 180
            angle = angle + (random.random() * 2 - 1) * self.rotate_limit

            shift_difference = self.max_pad - self.min_pad
            h_max_pad_ratio = random.random() * shift_difference + self.min_pad
            w_max_pad_ratio = random.random() * shift_difference + self.min_pad

            h_min_pad_ratio = random.random() * shift_difference + self.min_pad
            w_min_pad_ratio = random.random() * shift_difference + self.min_pad
        else:
            h_min_pad_ratio, w_max_pad_ratio, w_min_pad_ratio, h_max_pad_ratio = 0, 0, 0, 0
            angle = 0
            box = [[0, 0], [w, 0], [w, h], [0, h]]
        pad = [h_min_pad_ratio, w_min_pad_ratio, h_max_pad_ratio, w_max_pad_ratio]

        return {'angle': angle,
                'box': box,
                'pad': pad
                }

    @property
    def targets_as_params(self):
        return ['mask']

    def get_transform_init_args_names(self):
        return 'max_pad', 'min_pad', 'interpolation'


class CombineImagesAndCrop(DualTransform):
    """
        Crops horizontal strip, shifts and merges with prev/next digit image
        :param max_part_shift:
        :param min_part_shift:
        :param up: shifts up or down
    """

    def __init__(self, max_part_shift=40, min_part_shift=1, up=True, **kwargs):
        super().__init__(**kwargs)
        self.max_part_shift = max_part_shift
        self.min_part_shift = min_part_shift
        self.up = up

    def apply(self, img, prev_image, next_image, **params):
        length = random.randint(self.min_part_shift, self.max_part_shift)
        if self.up:
            return np.vstack((img[length:, :, :], next_image[:length, :, :]))
        else:
            return np.vstack((prev_image[img.shape[0] - length:, :, :], img[:img.shape[0] - length, :, :]))

    def get_params_dependent_on_targets(self, params):
        return {
            'prev_image': params['prev_image'],
            'next_image': params['next_image'],
        }

    @property
    def targets_as_params(self):
        return ['prev_image', 'next_image']


class RelativeRandomCrop(DualTransform):
    """Crop a random part of the random size of the input.

    Args:
        h_min_max_ratio ((float, float)): height crop size limits. This ratios must be in range [0, 1].
        w_min_max_ratio ((float, float)): width crop size limits. This ratios must be in range [0, 1].
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, h_min_max_ratio=(0, 1), w_min_max_ratio=(0, 1), interpolation=cv2.INTER_LINEAR,
                 always_apply=False, p=1.0):
        super(RelativeRandomCrop, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.h_min_max_ratio = h_min_max_ratio
        self.w_min_max_ratio = w_min_max_ratio

    def apply(self, img, x_min=0, y_min=0, x_max=0, y_max=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)

    def apply_to_bbox(self, bbox, x_min=0, y_min=0, x_max=0, y_max=0, **params):
        return F.bbox_crop(bbox, x_min, y_min, x_max, y_max, **params)

    def get_params_dependent_on_targets(self, params):
        image = params['image']
        h, w = image.shape[:2]
        h_min_ratio, h_max_ratio = self.h_min_max_ratio
        w_min_ratio, w_max_ratio = self.w_min_max_ratio

        crop_height = int(random.uniform(h_min_ratio, h_max_ratio) * h)
        crop_width = int(random.uniform(w_min_ratio, w_max_ratio) * w)

        y = random.randint(0, h - crop_height)
        x = random.randint(0, w - crop_width)

        return {'x_min': x,
                'y_min': y,
                'x_max': x + crop_width,
                'y_max': y + crop_height
                }

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return 'h_min_max_ratio', 'w_min_max_ratio', 'interpolation'


class ProportionalCenterCrop(DualTransform):
    def __init__(self, p_height_range, p_width_range, always_apply=False, p=1.0):
        super(ProportionalCenterCrop, self).__init__(always_apply, p)
        self.p_height_range = p_height_range
        self.p_width_range = p_width_range

    def apply(self, img, height=0, width=0, **params):
        return F.center_crop(img, height, width)

    def apply_to_bbox(self, bbox, height=0, width=0, **params):
        return F.bbox_random_crop(bbox, height, width, **params)

    def apply_to_keypoint(self, keypoint, height=0, width=0, **params):
        return F.keypoint_random_crop(keypoint, height, width, **params)

    def get_params_dependent_on_targets(self, params):
        image = params['image']
        h, w = image.shape[:2]
        crop_h = int(h * random.uniform(*self.p_height_range))
        crop_w = int(w * random.uniform(*self.p_width_range))

        return {'height': crop_h,
                'width': crop_w}

    def get_transform_init_args_names(self):
        return 'p_height_range', 'p_width_range'

    @property
    def targets_as_params(self):
        return ['image']


class ConditionalTranspose(DualTransform):
    """Transpose the input by swapping rows and columns if condition is True.

    Args:
        to_portrait (bool): transpose image to portrait if True else transpose to landscape
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, to_portrait=True, always_apply=False, p=1.0):
        super(ConditionalTranspose, self).__init__(always_apply, p)
        self.to_portrait = to_portrait

    def apply(self, img, apply_transpose=True, **params):
        if apply_transpose:
            img = F.transpose(img)
        return img

    def apply_to_bbox(self, bbox, apply_transpose=True, **params):
        if apply_transpose:
            bbox = F.bbox_transpose(bbox, 0, **params)
        return bbox

    def get_params_dependent_on_targets(self, params):
        image = params['image']
        h, w = image.shape[:2]
        apply_transpose = h < w if self.to_portrait else h > w

        return {'apply_transpose': apply_transpose}

    def get_transform_init_args_names(self):
        return 'to_portrait'

    @property
    def targets_as_params(self):
        return ['image']


class RelativePadIfNeeded(DualTransform):
    """Pad side of the image to desired width / height ratio.

    Args:
        p (float): probability of applying the transform. Default: 1.0.
        value (list of ints [r, g, b]): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int): padding value for mask if border_mode is cv2.BORDER_CONSTANT.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, w2h_ratio=1, border_mode=cv2.BORDER_REFLECT_101,
                 value=None, mask_value=None, always_apply=False, p=1.0):
        super(RelativePadIfNeeded, self).__init__(always_apply, p)
        self.w2h_ratio = w2h_ratio
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def get_params_dependent_on_targets(self, params):
        image = params['image']
        height, width = image.shape[:2]
        if width / height > self.w2h_ratio:
            min_width = width
            min_height = int(width / self.w2h_ratio)
        else:
            min_width = int(height * self.w2h_ratio)
            min_height = height

        if height < min_height:
            h_pad_top = int((min_height - height) / 2.0)
            h_pad_bottom = min_height - height - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if width < min_width:
            w_pad_left = int((min_width - width) / 2.0)
            w_pad_right = min_width - width - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        params.update({'pad_top': h_pad_top,
                       'pad_bottom': h_pad_bottom,
                       'pad_left': w_pad_left,
                       'pad_right': w_pad_right})
        return params

    def apply(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                 border_mode=self.border_mode, value=self.value)

    def apply_to_mask(self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        return F.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right,
                                 border_mode=self.border_mode, value=self.mask_value)

    def apply_to_bbox(self, bbox, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, rows=0, cols=0, **params):
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        bbox = [x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top]
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(self, keypoint, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params):
        x, y, a, s = keypoint
        return [x + pad_left, y + pad_top, a, s]

    def get_transform_init_args_names(self):
        return 'min_height', 'min_width', 'border_mode', 'value', 'mask_value'

    @property
    def targets_as_params(self):
        return ['image']


class ProportionalRandomResizedCrop(DualTransform):
    """Crop a random part of the random size of the input with given proportions.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of image distortion with respect to the initial aspect ratio (which is set to 1).
        value (list of ints [r, g, b]): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32"""

    def __init__(self, height, width, scale=(0.08, 1.0), ratio=(1, 1), interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, p=1.0, always_apply=False):
        super(ProportionalRandomResizedCrop, self).__init__(always_apply, p)
        aspect_ratio = width / height
        ratio = (aspect_ratio * ratio[0], aspect_ratio * ratio[1])
        self.aug = Compose([RelativePadIfNeeded(aspect_ratio, border_mode, value, mask_value),
                            RandomResizedCrop(height, width, scale=scale, ratio=ratio, interpolation=interpolation)])

    def __call__(self, *args, **kwargs):
        return self.aug(*args, **kwargs)

    def get_transform_init_args_names(self):
        return 'height', 'width', 'scale', 'interpolation', 'border_mode', 'value', 'mask_value'


class FlipAndConcatTransform(DualTransform):
    """Vertically flip initial image and concatenate flipped with original one.
    Works only for image and masks.

    Args:
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, masks

    Image types:
        uint8, float32
    """

    def __init__(self, p=1., always_apply=True):
        super().__init__(always_apply, p)
        self.hflip = VerticalFlip(always_apply=True)

    def __call__(self, *args, **kwargs):
        new_kwargs = self.hflip(*args, **kwargs)
        res = {}
        for key, arg in new_kwargs.items():
            if arg is None:
                res[key] = None
                continue

            key_type = self._additional_targets[key] if key in self._additional_targets else key

            if key_type in ['image', 'mask']:
                res[key] = np.concatenate([kwargs[key], arg], axis=1)
            elif key_type in ['masks']:
                res[key] = [np.concatenate([mask, new_mask], axis=1)
                            for mask, new_mask in zip(kwargs[key], arg)]

        return res

    def add_targets(self, additional_targets):
        super().add_targets(additional_targets)
        self.hflip._additional_targets = additional_targets

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks
        }


class DeterminedFlip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        direction (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, direction, always_apply=False):
        super().__init__(always_apply, p=1)
        self.d = direction

    def apply(self, img, d=0, **params):
        return F.random_flip(img, d)

    def get_params(self):
        return {"d": self.d}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoints(self, keypoints, **params):
        return keypoints_flip(keypoints, **params)

    def get_transform_init_args_names(self):
        return ()


class DeterminedRotate90(DualTransform):
    """Randomly rotate the input by 90 degrees zero or more times.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, factor=0, always_apply=False):
        super().__init__(always_apply, p=1)
        self.factor = factor

    def apply(self, img, factor=0, **params):
        """
        Args:
            factor (int): number of times the input will be rotated by 90 degrees.
        """
        return np.ascontiguousarray(np.rot90(img, factor))

    def get_params(self):
        # Random int in the range [0, 3]
        return {"factor": self.factor}

    def apply_to_bbox(self, bbox, factor=0, **params):
        return F.bbox_rot90(bbox, factor, **params)

    def apply_to_keypoint(self, keypoint, factor=0, **params):
        return F.keypoint_rot90(keypoint, factor, **params)

    def get_transform_init_args_names(self):
        return ()
