import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union, Tuple

import imagesize
import pandas as pd
import torch
from albumentations import BaseCompose, Compose, BboxParams
from albumentations.core.bbox_utils import convert_bboxes_to_albumentations, \
    convert_bboxes_from_albumentations, filter_bboxes as alb_filter_bboxes
from albumentations.core.composition import BasicTransform
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

from torchok.constructor import DATASETS
from torchok.data.datasets.base import ImageDataset


@DATASETS.register_class
class DetectionDataset(ImageDataset):
    """A dataset for image detection task.

    .. csv-table:: Detection csv example.
        :header: image_path, bbox, label

        image1.png, [[217.62, 240.54, 38.99, 57.75], [1.0, 240.24, 346.63, 186.76]], [0, 1]
        image2.png, [[102.49, 118.47, 7.9, 17.31]], [2, 1]
        image3.png, [[253.21, 271.07, 59.59, 60.97], [257.85, 224.48, 44.13, 97.0]], [2, 0]
    """

    def __init__(self,
                 data_folder: Union[Path, str],
                 annotation_path: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_column: str = 'image_path',
                 input_dtype: str = 'float32',
                 bbox_column: str = 'bbox',
                 bbox_dtype: str = 'float32',
                 target_column: str = 'label',
                 target_dtype: str = 'long',
                 image_format: str = 'rgb',
                 rgba_layout_color: Union[int, Tuple[int, int, int]] = 0,
                 test_mode: bool = False,
                 bbox_format: str = 'coco',
                 min_area: float = 0.0,
                 min_visibility: float = 0.0):
        """Init DetectionDataset.

        Args:
            data_folder: Directory with all the images.
            annotation_path: Path to the pkl or csv file with image paths, bboxes and labels.
                Path to images must be under column `image_path`, bboxes must be under `bbox` column and bbox labels
                must be under `label` column.
                User can change column names, if the input_column, bbox_column or target_column is given.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_column: Column name containing paths to the images.
            input_dtype: Data type of the torch tensors related to the image.
            bbox_column: Column name containing list of bboxes for every image.
            bbox_dtype: Data type of the torch tensors related to the bboxes.
            target_column: Column name containing bboxes labels.
            target_dtype: Data type of the torch tensors related to the bboxes labels.
            image_format: format of images that will be returned from dataset. Can be `rgb`, `bgr`, `rgba`, `gray`.
            rgba_layout_color: color of the background during conversion from `rgba`.
            test_mode: If True, only image without labels will be returned.
            bbox_format: Bboxes format, for albumentations transform. Supports the following formats:
                pascal_voc - [x_min, y_min, x_max, y_max] = [98, 345, 420, 462]
                albumentations - [x_min, y_min, x_max, y_max] = [0.1531, 0.71875, 0.65625, 0.9625]
                coco - [x_min, y_min, width, height] = [98, 345, 322, 117]
                yolo - [x_center, y_center, width, height] = [0.4046875, 0.8614583, 0.503125, 0.24375]
            min_area: Value in pixels  If the area of a bounding box after augmentation becomes smaller than min_area,
                Albumentations will drop that box. So the returned list of augmented bounding boxes won't contain
                that bounding box.
            min_visibility: Value between 0 and 1. If the ratio of the bounding box area after augmentation to the area
                of the bounding box before augmentation becomes smaller than min_visibility,
                Albumentations will drop that box. So if the augmentation process cuts the most of the bounding box,
                that box won't be present in the returned list of the augmented bounding boxes.

        Raises:
            RuntimeError: if annotation_path is not in `pkl` or `csv` format.
        """
        super().__init__(
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            image_format=image_format,
            rgba_layout_color=rgba_layout_color,
            test_mode=test_mode
        )
        self.data_folder = Path(data_folder)

        self.input_column = input_column
        self.target_column = target_column
        self.target_dtype = target_dtype
        self.bbox_column = bbox_column
        self.bbox_dtype = bbox_dtype

        self.bbox_format = bbox_format

        if annotation_path.endswith('csv'):
            df = pd.read_csv(self.data_folder / annotation_path)
            df[self.bbox_column] = df[self.bbox_column].apply(json.loads)
            df[self.target_column] = df[self.target_column].apply(json.loads)
        elif annotation_path.endswith('pkl'):
            df = pd.read_pickle(self.data_folder / annotation_path)
        else:
            raise ValueError('Detection dataset error. Annotation path is not in `csv` or `pkl` format')

        self.image_paths = df.pop(self.input_column)

        if not self.test_mode:
            labels = [torch.tensor(row) for row in df[self.target_column]]
            bboxes = [torch.tensor(row) for row in df[self.bbox_column]]

            for i, (path, bbox, label) in enumerate(zip(self.image_paths, bboxes, labels)):
                width, height = imagesize.get(self.data_folder / path)
                if len(bbox):
                    bbox, label = self.filter_bboxes(bbox, label, height, width)
                    labels[i] = label
                    bboxes[i] = bbox

            self.lengths = torch.cumsum(torch.tensor([len(i) for i in labels]), dim=0)
            self.labels = torch.cat(labels)
            self.bboxes = torch.cat(bboxes)

        bbox_params = BboxParams(format=self.bbox_format, label_fields=['label'],
                                 min_area=min_area, min_visibility=min_visibility)

        if self.augment is not None:
            self.augment = Compose(self.augment.transforms, bbox_params=bbox_params)

        self.transform = Compose(self.transform.transforms, bbox_params=bbox_params)

    def filter_bboxes(self, bboxes: Tensor, labels: Tensor, rows: int, cols: int) -> [Tensor, Tensor]:
        """Filter empty bounding boxes.

        Args:
            bboxes: List of bounding box.
            labels: array of bbox labels
            rows: Image height.
            cols: Image width.

        Returns:
            numpy array of bounding boxes and numpy array of labels of these boxes.
        """
        lbox = torch.hstack([bboxes, labels[..., None]])
        alb_lbox = convert_bboxes_to_albumentations(lbox, self.bbox_format, rows, cols)
        alb_lbox_fixed = alb_filter_bboxes(alb_lbox, rows, cols)
        lbox_fixed = torch.tensor(convert_bboxes_from_albumentations(alb_lbox_fixed, self.bbox_format, rows, cols))
        return lbox_fixed[:, :4], lbox_fixed[:, 4]

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.image_paths)

    def get_raw(self, idx: int) -> dict:
        image_path = self.data_folder / self.image_paths.iloc[idx]
        image = self._read_image(image_path)
        sample = {'image': image, 'index': idx, 'orig_img_shape': torch.tensor(image.shape)}

        if not self.test_mode:
            region_end = self.lengths[idx]
            region_start = self.lengths[idx - 1] if idx else 0

            sample['label'] = self.labels[region_start:region_end]
            sample['bboxes'] = self.bboxes[region_start:region_end]

        sample = self._apply_transform(self.augment, sample)

        return sample

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['target'] - Target class or labels, dtype=target_dtype.
            sample['bboxes'] - Target bboxes, dtype=bbox_dtype.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        sample = self.get_raw(idx)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        if not self.test_mode:
            sample['label'] = torch.tensor(sample['label']).type(torch.__dict__[self.target_dtype])
            sample['bboxes'] = torch.tensor(sample['bboxes']).type(torch.__dict__[self.bbox_dtype]).reshape(-1, 4)

        return sample

    def collate_fn(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        new_batch = defaultdict(list)
        for i, elem in enumerate(batch):
            new_batch['bboxes'].append(elem.pop('bboxes'))
            new_batch['label'].append(elem.pop('label'))

        output = default_collate(batch)
        output.update(new_batch)
        return output
