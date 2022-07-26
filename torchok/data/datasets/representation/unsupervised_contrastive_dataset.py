from pathlib import Path
from typing import Union, Optional

import torch
import pandas as pd
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.data.datasets.base import ImageDataset


class UnsupervisedContrastiveDataset(ImageDataset):
    """A dataset for unsupervised contrastive task.

    One image is transformed twice so that they are positive to each other.

    .. csv-table:: UnsupervisedContrastive csv example
        :header: image_path
        
        cat_1.jpg
        dog_1.jpg
    """

    def __init__(self,
                 data_folder: str,
                 csv_path: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 image_dtype: str = 'float32',
                 csv_columns_mapping: dict = None,
                 grayscale: bool = False):
        """Init UnsupervisedContrastiveDataset.

        Args:
            data_folder: Directory with all the images.
            csv_path: Path to the csv file with path to images and annotations.
                Path to images must be under column `input_column`.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            image_dtype: data type of of the torch tensors related to the image.
            csv_columns_mapping: Matches mapping column names. Key - TorchOK column name, Value - csv column name.
                default value: {'image_path': 'image_path'}
            grayscale: if True image will be read as grayscale otherwise as RGB.
        """
        super().__init__(transform, augment, image_dtype, grayscale)
        self.__data_folder = Path(data_folder)
        self.__csv_path = csv_path
        self.__csv_columns_mapping = csv_columns_mapping if csv_columns_mapping is not None\
            else {'image_path': 'image_path'}
        self.__input_column = self.__csv_columns_mapping['image_path']
        self.__csv = pd.read_csv(self.__data_folder / self.__csv_path, dtype={self.__input_column: 'str'})

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image_0'] - Tensor, representing image after augmentations and transformations, dtype=image_dtype.
            sample['image_1'] - Tensor, representing image after augmentations and transformations, dtype=image_dtype.
            sample['index'] - Index.
        """
        record = self.__csv.iloc[idx]
        image_path = self.__data_folder / record[self.__input_column]
        image = self._read_image(image_path)
        sample = {'image': image}

        sample_0_transformed = self._apply_transform(self.augment, sample)['image']
        sample_1_transformed = self._apply_transform(self.augment, sample)['image']

        sample_0_augmented = self._apply_transform(self.transform, {'image': sample_0_transformed})
        sample_1_augmented = self._apply_transform(self.transform, {'image': sample_1_transformed})

        sample_0 = sample_0_augmented['image'].type(torch.__dict__[self._image_dtype])
        sample_1 = sample_1_augmented['image'].type(torch.__dict__[self._image_dtype])

        return {'image_0': sample_0, 'image_1': sample_1, 'index': idx}

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.__csv)

    @property
    def csv_columns_mapping(self) -> dict:
        """Column name matching."""
        return self.__csv_columns_mapping
