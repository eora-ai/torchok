from pathlib import Path
from typing import Any, Union, Optional, Dict

import cv2
import torch
import numpy as np
import pandas as pd
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.data.datasets.base import ImageDataset


class ImageSegmentationDataset(ImageDataset):
    """A dataset for image segmentation task.

    .. csv-table:: Segmentation csv example.
        :header: image_path, mask

        image1.png, mask1.png
        image2.png, mask2.png
        image3.png, mask3.png
    """
    def __init__(self,
                 data_folder: Union[Path, str],
                 csv_path: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 image_dtype: str = 'float32',
                 target_dtype: str = 'int64',
                 csv_columns_mapping: Dict[str, str] = None,
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init ImageSegmentationDataset.

        Args:
            data_folder: Directory with all the images.
            csv_path: Path to the csv file with path to images and masks.
                Path to images must be under column `image_path` and annotations must be under `mask` column.
                User can change column names, if the `csv_columns_mapping` is given.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            image_dtype: Data type of of the torch tensors related to the image.
            target_dtype: Data type of of the torch tensors related to the target.
            csv_columns_mapping: Matches mapping column names. Key - TorchOK column name, Value - csv column name.
                default value: {'image_path': 'image_path', 'target': 'mask'}
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
        """
        super().__init__(transform, augment, image_dtype, grayscale, test_mode)

        self.data_folder = Path(data_folder)
        self.target_dtype = target_dtype
        self.csv_path = csv_path
        self.csv_columns_mapping = csv_columns_mapping if csv_columns_mapping is not None\
            else {'image_path': 'image_path',
                  'target': 'mask'}

        self.input_column = self.csv_columns_mapping['image_path']
        self.target_column = self.csv_columns_mapping['target']

        self.csv = pd.read_csv(self.data_folder / self.csv_path, dtype={self.input_column: 'str',
                                                                        self.target_column: 'str'})

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.csv.iloc[idx]
        image_path = self.data_folder / record[self.input_column]
        image = self._read_image(image_path)
        sample = {'image': image}

        if not self.test_mode:
            mask_path = self.data_folder / record[self.target_column]
            mask = self.__read_mask(mask_path)
            sample['mask'] = mask

        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)

        sample['image'] = sample['image'].type(torch.__dict__[self.image_dtype])
        sample['index'] = idx

        if not self.test_mode:
            sample['target'] = sample['mask'].type(torch.__dict__[self.target_dtype])
            del sample['mask']

        return sample

    def __read_mask(self, mask_path: str) -> np.ndarray:
        """Read mask.

        Args:
            mask_path: Path to mask.

        Raises:
            ValueError: If mask was not read correctly.
        """
        mask = cv2.imread(str(mask_path), 0)

        if mask is None:
            raise ValueError(f'{mask_path} was not read correctly!')

        return mask

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.csv)
