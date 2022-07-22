from abc import ABC, abstractmethod
from typing import Union, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose


class ImageDataset(Dataset, ABC):
    """An abstract class for image dataset."""

    def __init__(self,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 image_dtype: str = 'float32',
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init ImageDataset.

        Args:
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            image_dtype: Data type of the torch tensors related to the image.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
        """
        self._test_mode = test_mode
        self._transform = transform
        self._augment = augment
        self._image_dtype = image_dtype
        self._grayscale = grayscale


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
        image = cv2.imread(str(image_path), int(not self._grayscale))

        if image is None:
            raise ValueError(f'{image_path} image does not exist')
        if self._grayscale:
            image = image[..., None]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    @abstractmethod
    def __len__(self) -> int:
        """Dataset length."""
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> dict:
        """Get item sample."""
        pass

    @property
    def test_mode(self) -> bool:
        """Is test mode."""
        return self._test_mode

    @property
    def transform(self) -> Optional[Union[BasicTransform, BaseCompose]]:
        """Is transform to be applied on a sample."""
        return self._transform

    @property
    def augment(self) -> Optional[Union[BasicTransform, BaseCompose]]:
        """Is optional augment to be applied on a sample."""
        return self._augment

    @property
    def image_dtype(self) -> str:
        """Is data type of the torch tensors related to the image."""
        return self._image_dtype

    @property
    def grayscale(self) -> bool:
        """Is grayscale mode."""
        return self._grayscale
