from functools import partial
from pathlib import Path
from typing import Any, Optional, Union, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.constructor import DATASETS
from torchok.data.datasets.base import ImageDataset
from torchok.data.datasets.classification.classification import process_multilabel


@DATASETS.register_class
class MultiHeadImageDataset(ImageDataset):
    TARGET_TYPES = ['multiclass', 'multilabel', 'embedding']

    def __init__(self,
                 data_folder: str,
                 annotation_path: str,
                 targets: List[Dict[str, Any]],
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_column: str = 'image_path',
                 input_dtype: str = 'float32',
                 reader_library: str = 'opencv',
                 image_format: str = 'rgb',
                 rgba_layout_color: Union[int, Tuple[int, int, int]] = 0,
                 test_mode: bool = False,
                 lazy_init: bool = False):
        """Init ImageClassificationDataset.

        Args:
            data_folder: Directory with all the images.
            annotation_path: Path to the .pkl or .csv file with path to images and annotations.
                Path to images must be under column ``input_column`` and
                annotations must be under ``target_column`` column.
            targets: List of dicts where each dict contain information about heads

                - `name` (str): Name of the output target.
                - `column` (str): Column name containing image label.
                - `type` (str): Format of data processing. Available values: multiclass, multilabel, embedding.
                - `num_classes` (int): Number of classes.
                - `dtype` (str): Data type of the torch tensors related to the target.
                - `path_to_embeddings` (str): Used only when `target_type` is 'embedding',
                    path to .npy with embeddings per each image.

            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations`_ library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations`_ library.
            input_column: column name containing paths to the images.
            input_dtype: Data type of the torch tensors related to the image.
            reader_library: Image reading library. Can be 'opencv' or 'pillow'.
            image_format: format of images that will be returned from dataset. Can be `rgb`, `bgr`, `rgba`, `gray`.
            rgba_layout_color: color of the background during conversion from `rgba`.
            test_mode: If True, only image without labels will be returned.
            lazy_init: If True, for multilabel the target variable is converted to multihot when __getitem__ is called.
                For multiclass will check the class index to fit the range when ``__getitem__`` is called.
        .. _albumentations: https://albumentations.ai/docs/
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
        self.input_column = input_column
        self.annotation_path = annotation_path
        self.lazy_init = lazy_init

        if annotation_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_folder / annotation_path)
        elif annotation_path.endswith('.pkl'):
            self.df = pd.read_pickle(self.data_folder / annotation_path)
        else:
            raise ValueError('Detection dataset error. Annotation path is not in `csv` or `pkl` format')

        self.heads = []
        if not self.test_mode:
            for target in targets:
                name = target['name']
                column = target['column']
                target_type = target['type']
                num_classes = target.get('num_classes', None)
                target_dtype = target.get('dtype', 'long')

                if target_type == 'multiclass':
                    self.heads.append((name, column, target_type, num_classes, target_dtype))
                elif target_type == 'multilabel':
                    self.heads.append((name, column, target_type, num_classes, target_dtype))
                    self.df[column] = self.df[column].fillna('')
                    if not self.lazy_init:
                        self.df[column] = self.df[column].apply(partial(process_multilabel, num_classes=num_classes))
                elif target_type == 'embedding':
                    self.heads.append((name, column, target_type, num_classes, target_dtype))
                    data = np.load(self.data_folder / target['path_to_embeddings'], allow_pickle=True)
                    values = data[:, 1:].astype(self.target_type)
                    paths = data[:, 0]
                    df = pd.DataFrame({'image_path': paths, column: list(values)})
                    self.df = self.df.merge(df, on='image_path')
                else:
                    raise ValueError(f'This target {target_type} type is not supported')

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.df)

    def get_raw(self, idx: int):
        record = self.df.iloc[idx]
        image_path = self.data_folder / record[self.input_column]
        sample = {'image': self._read_image(image_path), 'index': idx}
        sample = self._apply_transform(self.augment, sample)

        if not self.test_mode:
            for name, column, target_type, num_classes, target_dtype in self.heads:
                label = record[column]
                if target_type == 'multilabel' and self.lazy_init:
                    label = process_multilabel(label, num_classes)
                sample[f'target_{name}'] = torch.tensor(label).type(torch.__dict__[target_dtype])

        return sample

    def __getitem__(self, idx: int) -> dict:
        sample = self.get_raw(idx)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        return sample
