from pathlib import Path
from typing import Any, Union, Optional, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.constructor import DATASETS
from torchok.data.datasets.base import ImageDataset


@DATASETS.register_class
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
                 input_column: str = 'image_path',
                 input_dtype: str = 'float32',
                 target_column: str = 'mask_path',
                 target_dtype: str = 'int64',
                 reader_library: str = 'opnecv',
                 image_format: str = 'rgb',
                 rgba_layout_color: Union[int, Tuple[int, int, int]] = 0,
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
            input_column: column name containing paths to the images.
            input_dtype: Data type of the torch tensors related to the image.
            target_dtype: Data type of the torch tensors related to the target.
            reader_library: Image reading library. Can be 'opnecv'or 'pillow'.
            image_format: format of images that will be returned from dataset. Can be `rgb`, `bgr`, `rgba`, `gray`.
            rgba_layout_color: color of the background during conversion from `rgba`.
            test_mode: If True, only image without labels will be returned.
        """
        super().__init__(
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            reader_library=reader_library,
            image_format=image_format,
            rgba_layout_color=rgba_layout_color,
            test_mode=test_mode
        )

        self.data_folder = Path(data_folder)
        self.csv_path = csv_path
        self.input_column = input_column
        self.target_column = target_column
        self.target_dtype = target_dtype

        self.csv = pd.read_csv(self.data_folder / self.csv_path, dtype={self.input_column: 'str',
                                                                        self.target_column: 'str'})

    def get_raw(self, idx: int) -> dict:
        record = self.csv.iloc[idx]
        image_path = self.data_folder / record[self.input_column]
        sample = {'image': self._read_image(image_path), 'index': idx}

        if not self.test_mode:
            mask_path = self.data_folder / record[self.target_column]
            sample['mask'] = self._read_mask(mask_path)

        sample = self._apply_transform(self.augment, sample)

        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.get_raw(idx)
        sample = self._apply_transform(self.transform, sample)

        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        if not self.test_mode:
            sample['target'] = sample.pop('mask').type(torch.__dict__[self.target_dtype])

        return sample

    def _read_mask(self, mask_path: str) -> np.ndarray:
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
