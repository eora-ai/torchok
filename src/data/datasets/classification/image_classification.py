from typing import Union, Optional
from functools import partial

import re
import torch
import numpy as np
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from src.registry import DATASETS
from src.data.datasets.base import ImageDataset


@DATASETS.register_class
class ImageClassificationDataset(ImageDataset):
    """A generic dataset for image classification task"""

    def __init__(self,
                 data_folder: str,
                 path_to_datalist: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 target_dtype: str = 'long',
                 target_column: str = 'label',
                 grayscale: bool = False,
                 expand_rate: int = 1,
                 test_mode: Optional[bool] = False,
                 multilabel: bool = False,
                 num_classes: int = None,
                 lazy_init_multilabel: bool = False):
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
        super().__init__(data_folder, path_to_datalist, transform, input_dtype, grayscale, test_mode, augment)

        self.__target_column = target_column
        self.__target_dtype = target_dtype
        self.__expand_rate = expand_rate
        self.__multilabel = multilabel

        self.update_transform_targets({'input': 'image'})

        if multilabel:
            self.__num_classes = num_classes
            self.__lazy_init_multilabel = lazy_init_multilabel
            if not self.lazy_init_multilabel and not self.test_mode:
                process_func = partial(self.process_multilabel, num_classes=self.num_classes)
                self.csv[self.target_column] = self.csv[self.target_column].apply(process_func)

    @property
    def target_column(self) -> str:
        return self.__target_column

    @property
    def target_dtype(self) -> str:
        return self.__target_dtype

    @property
    def expand_rate(self) -> int:
        return self.__expand_rate

    @property
    def multilabel(self) -> bool:
        return self.__multilabel

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @property
    def lazy_init_multilabel(self) -> bool:
        return self.__lazy_init_multilabel

    def __getitem__(self, idx: int) -> dict:
        idx = idx // self.expand_rate
        record = self.csv.iloc[idx]
        image = self._read_image(record)
        sample = {'input': image, 'index': idx}
        sample = self.apply_transform(self.augment, sample)
        sample = self.apply_transform(self.transform, sample)
        sample['input'] = sample['input'].type(torch.__dict__[self.input_dtype])

        if self.test_mode:
            return sample

        sample['target'] = record[self.target_column]

        if self.multilabel and self.lazy_init_multilabel:
            sample['target'] = self.process_multilabel(str(sample['target']), self.num_classes)
            sample['target'] = sample['target'].astype(self.input_dtype)

        sample['target'] = torch.tensor(sample['target']).type(torch.__dict__[self.target_dtype])

        return sample

    def __len__(self) -> int:
        csv_len = len(self.csv) * self.expand_rate
        return csv_len

    @staticmethod
    def process_multilabel(label: Union[int, str], num_classes: int) -> np.array:
        if isinstance(label, int):
            label = [label]
        else:
            label = list(map(int, re.findall(r'\d+', label)))

        max_label = max(label)

        if max_label < num_classes:
            multihot = np.zeros((num_classes,), dtype=bool)
            multihot[label] = True
        else:
            raise Exception(f'Target column contain label: {max_label}, it\'s more than num_classes = {num_classes}')
        return multihot
