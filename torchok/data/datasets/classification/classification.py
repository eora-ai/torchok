from pathlib import Path
from typing import Union, Optional

import re
import torch
import numpy as np
import pandas as pd
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.data.datasets.base import ImageDataset
from torchok.constructor import DATASETS


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
                 num_classes: int,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 image_dtype: str = 'float32',
                 target_dtype: str = 'long',
                 csv_columns_mapping: dict = None,
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
            num_classes: Number of classes (i.e. maximum class index in the dataset).
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations`_ library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations`_ library.
            image_dtype: Data type of of the torch tensors related to the image.
            target_dtype: Data type of of the torch tensors related to the target.
            csv_columns_mapping: Matches mapping column names. Key - TorchOK column name, Value - csv column name.
                default value: {'image_path': 'image_path', 'label': 'label'}
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
            multilabel: If True, targets are being converted to multihot vector for multilabel task.
                        If False, dataset prepares targets for multiclass classification.
            lazy_init: If True, the target variable is converted to multihot when ``__getitem__`` is called (multilabel).
                       For multiclass will check the class index to fit the range when ``__getitem__`` is called.

        .. _albumentations: https://albumentations.ai/docs/
        """
        super().__init__(transform, augment, image_dtype, grayscale, test_mode)

        self.__data_folder = Path(data_folder)
        self.__num_classes = num_classes
        self.__target_dtype = target_dtype
        self.__multilabel = multilabel
        self.__lazy_init = lazy_init
        self.__csv_path = csv_path
        self.__csv_columns_mapping = csv_columns_mapping if csv_columns_mapping is not None\
            else {'image_path': 'image_path',
                  'label': 'label'}

        self.__input_column = self.__csv_columns_mapping['image_path']
        self.__target_column = self.__csv_columns_mapping['label']

        if self.__multilabel:
            self.__csv = pd.read_csv(self.__data_folder / self.__csv_path, dtype={self.__input_column: 'str',
                                                                                  self.__target_column: 'str'})
            if not self.__lazy_init and not self.test_mode:
                self.__csv[self.__target_column] = self.__csv[self.__target_column].apply(self.__process_multilabel)
        else:
            self.__csv = pd.read_csv(self.__data_folder / self.__csv_path, dtype={self.__input_column: 'str',
                                                                                  self.__target_column: 'int'})
            if not self.__lazy_init and not self.test_mode:
                self.__csv[self.__target_column] = self.__csv[self.__target_column].apply(self.__process_multiclass)

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=image_dtype.
            sample['target'] - Target class or labels, dtype=target_dtype.
            sample['index'] - Index.
        """
        record = self.__csv.iloc[idx]
        image_path = self.__data_folder / record[self.__input_column]
        image = self._read_image(image_path)
        sample = {'image': image}
        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self._image_dtype])
        sample['index'] = idx

        if self._test_mode:
            return sample

        sample['target'] = record[self.__target_column]

        if self.__lazy_init:
            if self.__multilabel:
                sample['target'] = self.__process_multilabel(str(sample['target']))
                sample['target'] = sample['target'].astype(self.target_dtype)
            else:
                sample['target'] = self.__process_multiclass(sample['target'])

        sample['target'] = torch.tensor(sample['target']).type(torch.__dict__[self.__target_dtype])

        return sample

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.__csv)

    def __process_multiclass(self, class_idx: int) -> int:
        """Check the class index to fit the range.

        Args:
            class_idx: Target class index for multiclass classification.

        Returns:
            Verified class index.

        Raises:
            ValueError: If class index is out of range.
        """
        if class_idx >= self.__num_classes:
            raise ValueError(f'Target column contains class index: {class_idx}, '
                             f'it\'s more than num_classes = {self.__num_classes}')
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

        if max_label < self.__num_classes:
            multihot = np.zeros((self.__num_classes,), dtype=bool)
            multihot[labels] = True
        else:
            raise ValueError(f'Target column contains label: {max_label}, '
                             f'it\'s more than num_classes = {self.__num_classes}')
        return multihot

    @property
    def csv_columns_mapping(self) -> dict:
        """Column name matching."""
        return self.__csv_columns_mapping

    @property
    def target_dtype(self) -> str:
        """Is target type."""
        return self.__target_dtype

    @property
    def multilabel(self) -> bool:
        """Is task mode."""
        return self.__multilabel

    @property
    def num_classes(self) -> int:
        """Is number of classes."""
        return self.__num_classes

    @property
    def lazy_init(self) -> bool:
        """Is lazy init mode."""
        return self.__lazy_init
