from typing import Union, Optional

import re
import torch
import numpy as np
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from src.registry import DATASETS
from src.data.datasets.base import ImageDataset


@DATASETS.register_class
class ImageClassificationDataset(ImageDataset):
    """A generic dataset for multilabel/multiclass image classification task"""

    def __init__(self,
                 data_folder: str,
                 csv_path: str,
                 num_classes: int,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 target_dtype: str = 'long',
                 input_column: str = 'image_path',
                 target_column: str = 'label',
                 grayscale: bool = False,
                 test_mode: Optional[bool] = False,
                 multilabel: bool = False,
                 lazy_init: bool = False):
        """
        Args:
            data_folder: Directory with all the images.
            csv_path: Path to the csv file with path to images and annotations.
                Path to images must be under column `image_path` and annotations must be under `label` column
            num_classes: Number of classes (i.e. maximum class index in the dataset).
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of of the torch tensors related to the image.
            input_column: Name of the column that contains paths to images.
            target_dtype: Data type of of the torch tensors related to the target.
            target_column: Name of the column that contains target labels.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
            multilabel: If True, target labels are being converted to multihot vector for multilabel task.
                        If False, dataset prepare target for multiclass classification.
            lazy_init: If True, the target variable is converted to multihot when __getitem__ is called (multilabel).
                       For multiclass will check the class index to fit the range when __getitem__ is called.
        """
        super().__init__(data_folder, csv_path, transform, augment, input_dtype, input_column, grayscale, test_mode)

        self.__num_classes = num_classes
        self.__target_column = target_column
        self.__target_dtype = target_dtype
        self.__multilabel = multilabel
        self.__lazy_init = lazy_init

        if not self.__lazy_init and not self.test_mode:
            if multilabel:
                self.csv[self.target_column] = self.csv[self.target_column].apply(self.__process_multilabel)
            else:
                self.csv[self.target_column] = self.csv[self.target_column].apply(self.__process_multiclass)

    def __getitem__(self, idx: int) -> dict:
        record = self.csv.iloc[idx]
        image_path = record[self.input_column]
        image = self._read_image(image_path)
        sample = {'input': image, 'index': idx}
        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)
        sample['input'] = sample['input'].type(torch.__dict__[self.input_dtype])

        if self.test_mode:
            return sample

        sample['target'] = record[self.target_column]

        if self.lazy_init:
            if self.multilabel:
                sample['target'] = self.__process_multilabel(str(sample['target']))
                sample['target'] = sample['target'].astype(self.target_dtype)
            else:
                sample['target'] = self.__process_multiclass(str(sample['target']))

        sample['target'] = torch.tensor(sample['target']).type(torch.__dict__[self.target_dtype])

        return sample

    def __len__(self) -> int:
        return len(self.csv)

    def __process_multiclass(self, class_idx: Union[int, str]) -> int:
        """Check the class index to fit the range.

        Args:
            class_idx: Target class index for multiclass classification.

        Returns:
            Verified class index.

        Raises:
            ValueError: If class index is out of range.
        """
        class_idx = class_idx if isinstance(class_idx, int) else int(class_idx)
        if class_idx >= self.num_classes:
            raise ValueError(f'Target column contain class index: {class_idx}, ' +
                             f'it\'s more than num_classes = {self.num_classes}.')
        return class_idx

    def __process_multilabel(self, labels: str) -> np.array:
        """Convert label to multihot representation.

        Args:
            label: Target labels for multilabel classification.
                The class indexes must be separated by any separator.

        Returns:
            Multihot vector.

        Raises:
            ValueError: If class label is out of range.
        """
        labels = list(map(int, re.findall(r'\d+', labels)))

        max_label = max(labels)

        if max_label < self.num_classes:
            multihot = np.zeros((self.num_classes,), dtype=bool)
            multihot[labels] = True
        else:
            raise ValueError(f'Target column contain label: {max_label}, ' +
                             f'it\'s more than num_classes = {self.num_classes}.')
        return multihot

    @property
    def target_column(self) -> str:
        return self.__target_column

    @property
    def target_dtype(self) -> str:
        return self.__target_dtype

    @property
    def multilabel(self) -> bool:
        return self.__multilabel

    @property
    def num_classes(self) -> int:
        return self.__num_classes

    @property
    def lazy_init(self) -> bool:
        return self.__lazy_init
