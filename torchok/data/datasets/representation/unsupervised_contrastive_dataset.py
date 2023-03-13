from pathlib import Path
from typing import Optional, Union, Tuple
import warnings

import pandas as pd
import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.constructor import DATASETS
from torchok.data.datasets.base import ImageDataset


@DATASETS.register_class
class UnsupervisedContrastiveDataset(ImageDataset):
    """A dataset for unsupervised contrastive task.

    One image is transformed twice so that they are positive to each other.

    .. csv-table:: UnsupervisedContrastive csv example
        :header: image_path

        cat_1.jpg
        dog_1.jpg
    """

    def __init__(self,
                 data_folder: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 annotation_path: str = None,
                 input_column: str = 'image_path',
                 input_dtype: str = 'float32',
                 reader_library: str = 'opencv',
                 image_format: str = 'rgb',
                 rgba_layout_color: Union[int, Tuple[int, int, int]] = 0,
                 csv_path: str = None  # deprecated, will be removed later,
                 ):
        """Init UnsupervisedContrastiveDataset.

        Args:
            data_folder: Directory with all the images.
            annotation_path: Path to the .pkl or .csv file with path to images and annotations.
                Path to images must be under column `input_column`.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_column: column name containing paths to the images.
            input_dtype: data type of the torch tensors related to the image.
            reader_library: Image reading library. Can be 'opencv'or 'pillow'.
            image_format: format of images that will be returned from dataset. Can be `rgb`, `bgr`, `rgba`, `gray`.
            rgba_layout_color: color of the background during conversion from `rgba`.
            csv_path: DEPRECATED, Path to the .pkl or .csv file with path to images and annotations.
                Path to images must be under column `input_column`.
        """
        if annotation_path is None:
            if csv_path is not None:
                warnings.warn("`csv_path` is deprecated and will be removed in future version. "
                              "Use annotation_path instead.")
                annotation_path = csv_path
            else:
                raise ValueError("`annotation_path` must be specified.")

        super().__init__(
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            reader_library=reader_library,
            image_format=image_format,
            rgba_layout_color=rgba_layout_color,
        )
        self.data_folder = Path(data_folder)
        self.annotation_path = annotation_path
        self.input_column = input_column

        if annotation_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_folder / annotation_path, dtype={self.input_column: 'str'})
        elif annotation_path.endswith('.pkl'):
            self.df = pd.read_pickle(self.data_folder / annotation_path)
        else:
            raise ValueError('Detection dataset error. Annotation path is not in `csv` or `pkl` format')

    def get_raw(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image_0'] - Tensor, representing image after augmentations.
            sample['image_1'] - Tensor, representing image after augmentations.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        record = self.df.iloc[idx]
        image_path = self.data_folder / record[self.input_column]
        image = self._read_image(image_path)
        sample = {'image': image}

        sample_0 = self._apply_transform(self.augment, sample)['image']
        sample_1 = self._apply_transform(self.augment, sample)['image']

        return {'image_0': sample_0, 'image_1': sample_1, 'index': idx}

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image_0'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['image_1'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        sample = self.get_raw(idx)

        sample['image_0'] = self._apply_transform(self.transform, {'image': sample['image_0']})['image']
        sample['image_1'] = self._apply_transform(self.transform, {'image': sample['image_1']})['image']

        sample['image_0'] = sample['image_0'].type(torch.__dict__[self.input_dtype])
        sample['image_1'] = sample['image_1'].type(torch.__dict__[self.input_dtype])

        return sample

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.df)
