from ast import literal_eval
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from albumentations import BaseCompose, Compose, BboxParams
from albumentations.core.composition import BasicTransform

from torchok.constructor import DATASETS
from torchok.data.datasets.base import ImageDataset


@DATASETS.register_class
class DetectionDataset(ImageDataset):
    """
    """
    def __init__(self,
                 data_folder: str,
                 csv_path: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_column: str = 'image_path',
                 input_dtype: str = 'float32',
                 bbox_column: str = 'bbox',
                 bbox_dtype: str = 'float32',
                 target_column: str = 'label',
                 target_dtype: str = 'long',
                 grayscale: bool = False,
                 test_mode: bool = False,
                 bbox_format: str = 'coco',
                 min_area: float = 0.0,
                 min_visibility: float = 0.0,):

        super().__init__(
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            grayscale=grayscale,
            test_mode=test_mode
        )
        self.data_folder = Path(data_folder)
        self.csv = pd.read_csv(self.data_folder / csv_path)
        self.input_column = input_column
        
        self.target_column = target_column
        self.target_dtype = target_dtype

        self.bbox_column = bbox_column
        self.bbox_dtype = bbox_dtype

        self.bbox_format = bbox_format

        self.csv[self.bbox_column] = self.csv[self.bbox_column].apply(literal_eval)
        self.csv[self.target_column] = self.csv[self.target_column].apply(literal_eval)

        if self.augment is not None:
            self.augment = Compose(
                self.augment,
                bbox_params=BboxParams(
                    format=self.bbox_format,
                    label_fields=['label'],
                    min_area=min_area,
                    min_visibility=min_visibility
                )
            )

        self.transform = Compose(
            self.transform,
            bbox_params=BboxParams(
                format=self.bbox_format,
                label_fields=['label'],
                min_area=min_area,
                min_visibility=min_visibility
            )
        )

    def __len__(self) -> int:
        return len(self.csv)

    def get_raw(self, idx: int):
        record = self.csv.iloc[idx]
        image_path = self.data_folder / record[self.input_column]
        sample = {'image': self._read_image(image_path), 'index': idx}

        if not self.test_mode:
            target = record[self.target_column]
            bboxes = record[self.bbox_column]
            sample['label'] = target
            sample['bboxes'] = bboxes

        sample = self._apply_transform(self.augment, sample)

        return sample

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['target'] - Target class or labels, dtype=target_dtype.
            sample['bboxes'] - Target bboxes, dtype=bbox_dtype.
            sample['index'] - Index.
        """
        sample = self.get_raw(idx)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        if not self.test_mode:
            sample['label'] = torch.tensor(sample['label']).type(torch.__dict__[self.target_dtype])
            sample['bboxes'] = torch.tensor(sample['bboxes']).type(torch.__dict__[self.bbox_dtype])

        return sample
