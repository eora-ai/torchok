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
class ImageDataset(ABCDataset):
    """A generic dataset for image classification task"""

    def __init__(self,
                 data_folder: str,
                 path_to_datalist: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 target_dtype: str = 'long',
                 target_column: str = 'label',
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
        self.target_column = target_column
        self.input_dtype = input_dtype
        self.target_dtype = target_dtype
        self.expand_rate = expand_rate

        self.update_transform_targets({'input': 'image'})

    def __getitem__(self, idx: int) -> dict:
        sample = self.get_raw(idx)
        sample = self.apply_transform(self.transform, sample)
        sample['input'] = sample['input'].type(torch.__dict__[self.input_dtype])
        if not self.test_mode:
            sample["target"] = torch.tensor(sample["target"]).type(torch.__dict__[self.target_dtype])

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
            sample["target"] = record[self.target_column]

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


@DATASETS.register_class
class MultiLabelDataset(ImageDataset):
    """Dataset for multi label classification problem."""

    def __init__(self, num_classes: int, lazy_init: bool = False, **dataset_params):
        """
        Args:
            num_classes: Number of classes in multi label classification task.
                In csv classes must be presented as string containing list of indexes of sample's classes.
                Indexes may be separated any non-digit separator containing ascii characters.
                For example: `1, 3, 6` or `1-3-6` will be considered as [1, 3, 6]. Duplicating indices will be reduced.
                `1, 3, 6, 3` will be considered as [1, 3, 6].
            lazy_init: If True labels will be converted into one-hot vectors during the training otherwise
                they will be converted during the dataset initialization
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
        super().__init__(**dataset_params)
        self.num_classes = num_classes
        self.lazy_init = lazy_init
        if not lazy_init:
            process_func = partial(self.process_multilabel, num_classes=num_classes)
            self.csv[self.target_column] = self.csv[self.target_column].apply(process_func)

    def get_raw(self, idx: int) -> dict:
        sample = super().get_raw(idx)

        if not self.test_mode and self.lazy_init:
            sample["target"] = self.process_multilabel(str(sample["target"]), self.num_classes)
            sample["target"] = sample["target"].astype(self.input_dtype)
        return sample

    @staticmethod
    def process_multilabel(label, num_classes):
        if isinstance(label, int):
            label = [label]
        else:
            label = list(map(int, re.findall('\d+', label)))
        multihot = np.zeros((num_classes,), dtype=bool)
        multihot[label] = True

        return multihot


@DATASETS.register_class
class MultiHeadImageDataset(ImageDataset):
    TARGET_TYPES = ['multiclass', 'multilabel', 'feature_label']

    def __init__(self, targets, lazy_init: bool = False, **dataset_params):
        super().__init__(**dataset_params)
        self.lazy_init = lazy_init
        self.heads = []
        if not self.test_mode:
            for target in targets:
                column = target['column']
                name = target['name']
                target_type = target['target_type']
                num_classes = target['num_classes']

                if target_type in range(len(self.TARGET_TYPES)):
                    target_type = self.TARGET_TYPES[target_type]

                if target_type == 'multiclass':
                    self.heads.append((name, target_type, num_classes, column))
                elif target_type == 'multilabel':
                    self.heads.append((name, target_type, num_classes, column))
                    self.csv[column] = self.csv[column].fillna('')
                    if not self.lazy_init:
                        process_func = partial(MultiLabelDataset.process_multilabel, num_classes=num_classes)
                        self.csv[column] = self.csv[column].apply(process_func)
                elif target_type == 'feature_label':
                    self.heads.append((name, target_type, num_classes, column))
                    data = np.load(self.data_folder / target['path_to_labels'], allow_pickle=True)
                    values = data[:, 1:].astype(self.input_dtype)
                    paths = data[:, 0]
                    df = pd.DataFrame({'image_path': paths, column: list(values)})
                    self.csv = self.csv.merge(df, on='image_path')
                else:
                    raise ValueError(f'This target {target_type} type is not supported')

    def get_raw(self, idx: int):
        record = self.csv.iloc[idx]
        image = self.read_image(record)
        sample = {"input": image, 'index': idx}
        sample = self.apply_transform(self.augment, sample)

        if not self.test_mode:
            for name, target_type, num_classes, label_name in self.heads:
                label = record[label_name]
                if target_type == 'multilabel':
                    if self.lazy_init:
                        label = MultiLabelDataset.process_multilabel(label, num_classes)
                    label = label.astype(self.input_dtype)
                sample[f'target_{name}'] = label

        return sample

    def __getitem__(self, idx: int) -> dict:
        sample = self.get_raw(idx)
        sample = self.apply_transform(self.transform, sample)
        sample['input'] = sample['input'].type(torch.__dict__[self.input_dtype])

        return sample
