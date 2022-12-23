import re
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.constructor import DATASETS
from torchok.data.datasets.base import ImageDataset


@DATASETS.register_class
class ImageClassificationDataset(ImageDataset):
    """A generic dataset for multilabel/multiclass image classification task.

    .. csv-table:: Multiclass task csv example.
        :header: image_path, label

        cat_1.jpg, 1
        dog_1.jpg, 0

    .. csv-table:: Multilabel task csv example.
        :header: image_path, label

        cat_dog_1.jpg, 0 1
        cat_dog_2.jpg, 0 1
        dog_1.jpg, 0
    """

    def __init__(self,
                 data_folder: str,
                 csv_path: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 num_classes: int = None,
                 input_column: str = 'image_path',
                 input_dtype: str = 'float32',
                 target_column: str = 'label',
                 target_dtype: str = 'long',
                 channel_order: str = 'rgb',
                 grayscale: bool = False,
                 test_mode: bool = False,
                 multilabel: bool = False,
                 lazy_init: bool = False):
        """Init ImageClassificationDataset.

        Args:
            data_folder: Directory with all the images.
            csv_path: Path to the csv file with path to images and annotations.
                Path to images must be under column ``input_column`` and
                annotations must be under ``target_column`` column.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations`_ library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations`_ library.
            num_classes: Number of classes (i.e. maximum class index in the dataset).
            input_column: column name containing paths to the images.
            input_dtype: Data type of the torch tensors related to the image.
            target_column: column name containing image label.
            target_dtype: Data type of the torch tensors related to the target.
            channel_order: Order of channel, candidates are `bgr` and `rgb`.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
            multilabel: If True, targets are being converted to multihot vector for multilabel task.
                        If False, dataset prepares targets for multiclass classification.
            lazy_init: If True, for multilabel the target variable is converted to multihot when __getitem__ is called.
                For multiclass will check the class index to fit the range when ``__getitem__`` is called.

        .. _albumentations: https://albumentations.ai/docs/
        """
        super().__init__(
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            channel_order=channel_order,
            grayscale=grayscale,
            test_mode=test_mode
        )

        if num_classes is None and multilabel:
            raise ValueError('``num_classes`` must be specified when ``multilabel`` is `True`')

        self.data_folder = Path(data_folder)
        self.num_classes = num_classes
        self.input_column = input_column
        self.target_column = target_column
        self.target_dtype = target_dtype
        self.multilabel = multilabel
        self.lazy_init = lazy_init
        self.csv_path = csv_path

        csv_path = self.data_folder / self.csv_path
        dtype = {self.input_column: 'str', self.target_column: 'str' if self.multilabel else 'int'}

        self.csv = pd.read_csv(csv_path, dtype=dtype)
        if not self.lazy_init and not self.test_mode:
            self.csv[self.target_column] = self.csv[self.target_column].apply(self.process_function)

    def get_raw(self, idx: int) -> dict:
        """Get item sample without transform application.

        Returns:
            sample: dict, where
            sample['image'] - np.array, representing image after augmentations.
            sample['target'] - Target class or labels.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        record = self.csv.iloc[idx]
        image_path = self.data_folder / record[self.input_column]
        sample = {'image': self._read_image(image_path), 'index': idx}

        if not self.test_mode:
            target = record[self.target_column]
            if self.lazy_init:
                target = self.process_function(target)
            sample['target'] = target

        sample = self._apply_transform(self.augment, sample)

        return sample

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['target'] - Target class or labels, dtype=target_dtype.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        sample = self.get_raw(idx)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        if not self.test_mode:
            sample['target'] = torch.tensor(sample['target']).type(torch.__dict__[self.target_dtype])

        return sample

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.csv)

    def process_function(self, target: Any) -> Any:
        """Prepare dataset target based of classification type.

        Args:
            target: Classification labels to prepare.

        Returns:
            Prepared classification labels.
        """
        if self.multilabel:
            return self.__process_multilabel(target)
        else:
            return self.__process_multiclass(target)

    def __process_multiclass(self, class_idx: int) -> int:
        """Check the class index to fit the range.

        Args:
            class_idx: Target class index for multiclass classification.

        Returns:
            Verified class index.

        Raises:
            ValueError: If class index is out of range.
        """
        if self.num_classes is not None and class_idx >= self.num_classes:
            raise ValueError(f'Target column contains class index: {class_idx}, '
                             f'it\'s more than num_classes = {self.num_classes}')
        return class_idx

    def __process_multilabel(self, labels: str) -> np.array:
        """Convert label to multihot representation.

        Args:
            labels: Target labels for multilabel classification.
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
            raise ValueError(f'Target column contains label: {max_label}, '
                             f'it\'s more than num_classes = {self.num_classes}')
        return multihot
