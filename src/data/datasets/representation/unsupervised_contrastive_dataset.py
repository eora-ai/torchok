from typing import Union, Optional

import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from src.registry import DATASETS
from src.data.datasets.base import ImageDataset


@DATASETS.register_class
class UnsupervisedContrastiveDataset(ImageDataset):
    """A generic dataset for image contrastive task"""

    def __init__(self,
                 data_folder: str,
                 path_to_datalist: str,
                 transform: Union[BasicTransform, BaseCompose],
                 input_dtype: str = 'float32',
                 grayscale: bool = False,
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None):
        """
        Args:
            data_folder: Directory with all the images.
            path_to_datalist: Path to the csv file with path to images and annotations.
                Path to images must be under column `image_path` and annotations must be under `label` column
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            input_dtype: data type of of the torch tensors related to the image
            grayscale: if True image will be read as grayscale otherwise as RGB.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
        """
        super().__init__(data_folder, path_to_datalist, transform, input_dtype, grayscale, augment=augment)

        self.update_transform_targets({'input': 'image'})

    def __getitem__(self, idx: int) -> dict:
        record = self.csv.iloc[idx]
        image = self._read_image(record)
        sample = {'input': image}
        sample_0_transformed = self.apply_transform(self.augment, sample)['input']
        sample_1_transformed = self.apply_transform(self.augment, sample)['input']

        sample_0_augmented = self.apply_transform(self.transform, {'input': sample_0_transformed})
        sample_1_augmented = self.apply_transform(self.transform, {'input': sample_1_transformed})

        sample_0 = sample_0_augmented['input'].type(torch.__dict__[self.input_dtype])
        sample_1 = sample_1_augmented['input'].type(torch.__dict__[self.input_dtype])

        return {'input_0': sample_0, 'input_1': sample_1, 'index': idx}

    def __len__(self) -> int:
        return len(self.csv)
