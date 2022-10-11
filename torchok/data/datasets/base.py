from abc import ABC, abstractmethod
from typing import Optional, Union

import cv2
import numpy as np
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torch.utils.data import Dataset


class ImageDataset(Dataset, ABC):
    """An abstract class for image dataset."""

    def __init__(self,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 channel_order: str = 'rgb',
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init ImageDataset.

        Args:
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of the torch tensors related to the image.
            channel_order: Order of channel, candidates are `bgr` and `rgb`.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
        """
        self.test_mode = test_mode
        self.transform = transform
        self.augment = augment
        self.input_dtype = input_dtype
        self.grayscale = grayscale
        self.channel_order = channel_order

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
        image = cv2.imread(str(image_path), int(not self.grayscale))

        if image is None:
            raise ValueError(f'{image_path} image does not exist')

        if self.grayscale:
            image = image[..., None]
        elif self.channel_order == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
