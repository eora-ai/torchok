from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from src.registry import DATASETS
from .abc_dataset import ABCDataset


@DATASETS.register_class
class ContrastiveDataset(ABCDataset):
    """A generic dataset for image classification task"""

    def __init__(self,
                 data_folder: str,
                 path_to_datalist: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 grayscale: bool = False):
        """
        Args:
            data_folder: Directory with all the images.
            path_to_datalist: Path to the csv file with path to images and annotations.
                Path to images must be under column `image_path` and annotations must be under `label` column
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: data type of of the torch tensors related to the image
            grayscale: if True image will be read as grayscale otherwise as RGB.
        """
        super().__init__(transform, augment)
        self.data_folder = Path(data_folder)
        self.csv = pd.read_csv(self.data_folder / path_to_datalist)

        self.grayscale = grayscale
        self.input_dtype = input_dtype

        self.update_transform_targets({'input': 'image'})

    def __getitem__(self, idx: int) -> dict:
        sample = self.get_raw(idx)
        sample_0 = self.apply_transform(self.transform, {'input': sample['input_0']})
        sample_1 = self.apply_transform(self.transform, {'input': sample['input_1']})
        sample_0 = sample_0['input'].type(torch.__dict__[self.input_dtype])
        sample_1 = sample_1['input'].type(torch.__dict__[self.input_dtype])

        return {'input_0': sample_0, 'input_1': sample_1, 'index': idx}

    def __len__(self) -> int:
        return len(self.csv)

    def get_raw(self, idx: int) -> dict:
        record = self.csv.iloc[idx]
        image = self.read_image(record)
        sample = {"input": image}
        sample_0 = self.apply_transform(self.augment, sample)['input']
        sample_1 = self.apply_transform(self.augment, sample)['input']

        return {'input_0': sample_0, 'input_1': sample_1, 'index': idx}

    def read_image(self, record) -> np.ndarray:
        image_path = self.data_folder / record.image_path
        image = cv2.imread(str(image_path), int(not self.grayscale))

        if image is None:
            raise ValueError(f'{image_path} image does not exist')
        if self.grayscale:
            image = image[..., None]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
