from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple

import cv2
import numpy as np
from PIL.Image import open as imopen
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torch.utils.data import Dataset


class ImageDataset(Dataset, ABC):
    """An abstract class for image dataset."""

    def __init__(self,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 image_format: str = 'rgb',
                 rgba_layout_color: Union[int, Tuple[int, int, int]] = 0,
                 test_mode: bool = False):
        """Init ImageDataset.

        Args:
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of the torch tensors related to the image.
            image_format: format of images that will be returned from dataset. Can be `rgb`, `bgr`, `rgba`, `gray`.
            rgba_layout_color: color of the background during conversion from `rgba`.
            test_mode: If True, only image without labels will be returned.
        """
        self.test_mode = test_mode
        self.transform = transform
        self.augment = augment
        self.input_dtype = input_dtype
        self.image_format = image_format
        self.rgba_layout_color = rgba_layout_color

    def _apply_transform(self, transform: Union[BasicTransform, BaseCompose], sample: dict) -> dict:
        """Is transformations based on API of albumentations library.

        Args:
            transform: Transformations from `albumentations` library.
                https://github.com/albumentations-team/albumentations/
            sample: Sample which the transformation will be applied to.

        Returns:
            Transformed sample.
        """
        if transform is None:
            return sample

        new_sample = transform(**sample)
        return new_sample

    def _read_image(self, image_path: str) -> np.ndarray:
        image = np.array(imopen(image_path))

        if self.image_format == 'rgb':
            if image.ndim == 2:  # Gray
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                alpha = image[..., 3:4] / 255
                image = np.clip(image[..., :3] * alpha + self.rgba_layout_color * (1 - alpha), a_min=0, a_max=255)
                image = image.astype('uint8')
            elif image.shape[2] == 2:
                gray = image[..., 0]
                alpha = image[..., 1] / 255
                image = (gray * alpha + self.rgba_layout_color * (1 - alpha)).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif self.image_format == 'rgba':
            if image.ndim == 2:  # Gray
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            elif image.shape[2] == 2:
                gray = image[..., 0]
                alpha = image[..., 1] / 255
                image = (gray * alpha + self.rgba_layout_color * (1 - alpha)).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        elif self.image_format == 'bgr':
            if image.ndim == 2:  # Gray
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # RGBA
                alpha = image[..., 3:4] / 255
                image = np.clip(image[..., :3] * alpha + self.rgba_layout_color * (1 - alpha), a_min=0, a_max=255)
                image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 2:
                gray = image[..., 0]
                alpha = image[..., 1] / 255
                image = (gray * alpha + self.rgba_layout_color * (1 - alpha)).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif self.image_format == 'gray':
            if image.ndim == 3 and image.shape[2] == 4:  # RGBA
                alpha = image[..., 3:4] / 255
                image = np.clip(image[..., :3] * alpha + self.rgba_layout_color * (1 - alpha), a_min=0, a_max=255)
                image = image.astype('uint8')

            if image.ndim == 3 and image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            if image.ndim == 3 and image.shape[2] == 2:
                gray = image[..., 0]
                alpha = image[..., 1] / 255
                image = np.clip(gray * alpha + self.rgba_layout_color * (1 - alpha), a_min=0, a_max=255)

            if image.ndim == 2:  # Gray
                image = image[..., None]
        else:
            raise ValueError(f'Unsupported image format `{self.image_format}`')

        return image

    @abstractmethod
    def __len__(self) -> int:
        """Dataset length."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Get item sample."""
        pass

    @abstractmethod
    def get_raw(self, idx: int) -> dict:
        """Get item sample without transform application."""
        pass
