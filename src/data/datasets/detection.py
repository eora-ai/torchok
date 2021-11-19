import cv2
import numpy as np
import pandas as pd
import torch

from src.registry import DATASETS
from .classification import ImageDataset


@DATASETS.register_class
class ImageDetectionDataset(ImageDataset):
    def __init__(self, 
                target_cls_column='class_label',
                target_bbox_columns=['x_c', 'y_c', 'w', 'h'],
                **dataset_params
     ):
        super().__init__(**dataset_params)
        

    #     transform_targets = {'input': 'image', 'target': 'mask'}
    #     self.update_transform_targets(transform_targets)

    # def __getitem__(self, idx: int):
    #     sample = self.get_raw(idx // self.expand_rate)
    #     sample = self.apply_transform(self.transform, sample)
    #     sample['input'] = sample['input'].type(torch.__dict__[self.input_dtype])
    #     if not self.test_mode:
    #         sample['target'] = sample['target'].type(torch.__dict__[self.target_dtype])
    #     return sample

    # def get_raw(self, idx: int):
    #     idx = idx // self.expand_rate
    #     record = self.csv.iloc[idx]
    #     image = self.read_image(record)
    #     sample = {'input': image, 'index': idx, 'shape': torch.tensor(image.shape[:2])}
    #     if not self.test_mode:
    #         mask = self.read_mask(record[self.target_column])
    #         if mask is None:
    #             mask = np.zeros(image.shape[:2], dtype='uint8')
    #         sample["target"] = mask
    #     sample = self.apply_transform(self.augment, sample)

    #     return sample

    # def read_mask(self, rel_path):
    #     if isinstance(rel_path, float):
    #         return None
    #     mask_path = self.data_folder / rel_path
    #     mask = cv2.imread(str(mask_path), 0)
    #     if mask is None:
    #         raise ValueError(f'{mask_path} was not read correctly!')
    #     return mask
