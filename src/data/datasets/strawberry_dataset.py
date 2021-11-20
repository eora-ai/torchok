import re
from functools import partial
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
class StrawbettyDataset(ABCDataset):
    """A generic dataset for image classification task"""

    def __init__(self,
                 data_folder: str,
                 path_to_datalist: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 target_cls_dtype: str = 'long',
                 target_cls_column: str = 'label',
                 target_reg_dtype:str = 'float32',
                 target_reg_column = 'healthy',
                 grayscale: bool = False,
                 expand_rate: int = 1,
                 test_mode: Optional[bool] = False):
        """
        Args:
            data_folder: Directory with all the images.
            path_to_datalist: Path to the csv file with path to images and annotations.
                Path to images must be under column `image_path` and annotations must be under `label` column
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of of the torch tensors related to the image.
            target_dtype: Data type of of the torch tensors related to the target.
            target_column: Name of the column that contains target labels.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            expand_rate: A multiplier that shows how many times the dataset will be larger than its real size.
                Useful for small datasets.
            test_mode: If True, only image without labels will be returned.
        """
        super().__init__(transform, augment)
        self.data_folder = Path(data_folder)
        self.csv = pd.read_csv(self.data_folder / path_to_datalist)

        self.grayscale = grayscale
        self.test_mode = test_mode
        self.target_cls_column = target_cls_column
        self.target_reg_column = target_reg_column
        self.input_dtype = input_dtype
        self.target_cls_dtype = target_cls_dtype
        self.target_reg_dtype = target_reg_dtype
        self.expand_rate = expand_rate

        self.update_transform_targets({'input': 'image'})

    def __getitem__(self, idx: int) -> dict:
        sample = self.get_raw(idx)
        sample = self.apply_transform(self.transform, sample)
        sample['input'] = sample['input'].type(torch.__dict__[self.input_dtype])
        if not self.test_mode:
            sample["cls_target"] = torch.tensor(sample["cls_target"]).type(torch.__dict__[self.target_cls_dtype])
            sample["reg_target"] = torch.tensor(sample["reg_target"]).type(torch.__dict__[self.target_reg_dtype])
        return sample

    def __len__(self) -> int:
        return len(self.csv) * self.expand_rate

    def get_raw(self, idx: int) -> dict:
        idx = idx // self.expand_rate
        record = self.csv.iloc[idx]
        image = self.read_image(record)
        sample = {"input": image, 'index': idx}
        sample = self.apply_transform(self.augment, sample)

        if not self.test_mode:
            sample["cls_target"] = record[self.target_cls_column]
            sample["reg_target"] = record[self.target_reg_column]
        return sample

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

