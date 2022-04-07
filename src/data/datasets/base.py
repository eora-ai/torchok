from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union, Optional

import cv2
import numpy as np
import pandas as pd
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torch.utils.data import Dataset


class ImageDataset(Dataset, ABC):
    """ An abstract class for image dataset """

    def __init__(self,
                 data_folder: str,
                 csv_path: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 input_column: str = 'image_path',
                 grayscale: bool = False,
                 test_mode: bool = False,
                 transform_targets: dict = {'input': 'image'}):
        """
        Args:
            data_folder: Directory with all the images.
            csv_path: Path to the csv file with path to images and annotations.
                Path to images must be under column `image_path` and annotations must be under `label` column.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of of the torch tensors related to the image.
            input_column: Name of the column that contains paths to images.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
            transform_target: Transformations target for `albumentations` library.
        """
        self.__test_mode = test_mode
        self.__transform = transform
        self.__augment = augment
        self.__input_dtype = input_dtype
        self.__input_column = input_column
        self.__grayscale = grayscale
        self.__transform_targets = transform_targets

        self.__data_folder = Path(data_folder)
        self.__csv = pd.read_csv(self.data_folder / csv_path)
        self.__update_transform_targets(transform_targets)

    def _apply_transform(self, transform: Union[BasicTransform, BaseCompose], sample: dict) -> dict:
        """Apply transformations (from `albumentations` library) to given sample.

        Args:
            transform: Transformations from `albumentations` library.
                https://github.com/albumentations-team/albumentations/
            sample: Sample to which the transformation will be applied.

        Returns:
            Transformed sample.
        """
        if transform is None:
            return sample

        new_sample = {}
        # mapping to `albumentations` acceptable parameters
        for source, target in self.transform_targets.items():
            if source in sample:
                if source == 'input' or source == 'target':
                    new_sample[target] = sample[source]
                else:
                    new_sample[source] = sample[source]

        new_sample = transform(**new_sample)
        # inverse mapping
        for source, target in self.transform_targets.items():
            if target in new_sample and (source == 'input' or source == 'target'):
                sample[source] = new_sample[target]
            elif source in new_sample:
                sample[source] = new_sample[source]
        return sample

    def _read_image(self, image_path: str) -> np.ndarray:
        full_image_path = self.data_folder / image_path
        image = cv2.imread(str(full_image_path), int(not self.grayscale))

        if image is None:
            raise ValueError(f'{image_path} image does not exist')
        if self.grayscale:
            image = image[..., None]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __update_transform_targets(self, transform_targets: dict) -> None:
        self.transform.additional_targets = transform_targets
        self.transform.add_targets(transform_targets)
        if self.augment is not None:
            self.augment.additional_targets = transform_targets
            self.augment.add_targets(transform_targets)

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> dict:
        pass

    @property
    def test_mode(self) -> bool:
        return self.__test_mode

    @property
    def transform(self) -> Optional[Union[BasicTransform, BaseCompose]]:
        return self.__transform

    @property
    def augment(self) -> Optional[Union[BasicTransform, BaseCompose]]:
        return self.__augment

    @property
    def input_dtype(self) -> str:
        return self.__input_dtype

    @property
    def input_column(self) -> str:
        return self.__input_column

    @property
    def grayscale(self) -> bool:
        return self.__grayscale

    @property
    def transform_targets(self) -> dict:
        return self.__transform_targets

    @property
    def data_folder(self) -> Path:
        return self.__data_folder

    @property
    def csv(self) -> pd.DataFrame:
        return self.__csv
