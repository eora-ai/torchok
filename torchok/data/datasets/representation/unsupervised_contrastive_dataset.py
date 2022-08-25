from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
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
                 input_column: str = 'image_path',
                 input_dtype: str = 'float32',
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
            input_column: column name containing paths to the images.
            input_dtype: data type of the torch tensors related to the image.
            grayscale: if True image will be read as grayscale otherwise as RGB.
        """
        super().__init__(transform, augment, input_dtype, grayscale)
        self.data_folder = Path(data_folder)
        self.csv_path = csv_path
        self.input_column = input_column
        self.csv = pd.read_csv(self.data_folder / self.csv_path, dtype={self.input_column: 'str'})

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image_0'] - Tensor, representing image after augmentations and transformations, dtype=image_dtype.
            sample['image_1'] - Tensor, representing image after augmentations and transformations, dtype=image_dtype.
            sample['index'] - Index.
        """
        record = self.csv.iloc[idx]
        image_path = self.data_folder / record[self.input_column]
        image = self._read_image(image_path)
        sample = {'image': image}

        sample_0_transformed = self._apply_transform(self.augment, sample)['image']
        sample_1_transformed = self._apply_transform(self.augment, sample)['image']

        sample_0_augmented = self._apply_transform(self.transform, {'image': sample_0_transformed})
        sample_1_augmented = self._apply_transform(self.transform, {'image': sample_1_transformed})

        sample_0 = sample_0_augmented['image'].type(torch.__dict__[self.input_dtype])
        sample_1 = sample_1_augmented['image'].type(torch.__dict__[self.input_dtype])

        return {'image_0': sample_0, 'image_1': sample_1, 'index': idx}

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.csv)
