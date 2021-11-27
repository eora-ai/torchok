import cv2
import numpy as np
import pandas as pd
import torch
from typing import List

from src.data import transforms as module_transforms
from torch.utils.data._utils.collate import default_collate

from src.registry import DATASETS
from .classification import ImageDataset



@DATASETS.register_class
class DetectionDataset(ImageDataset):
    """
    DetectionDataset class annotation_format:
    [x_min, y_min, x_max, y_max, label] - pascal_voc format in albumentation see the link
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

    Example:
    [{'x_min': 520, 'y_min': 148, 'x_max': 600, 'y_max': 201, 'label': 20},
     {'x_min': 598, 'y_min': 206, 'x_max': 675, 'y_max': 240, 'label': 1}]
    """
    def __init__(self, target_column='annotation', **dataset_params):
        super().__init__(**dataset_params)
        
        self.target_column = target_column
        if self.augment is not None:
            self.augment = module_transforms.Compose(
                self.augment,
                bbox_params=module_transforms.BboxParams(
                    format='pascal_voc',
                    label_fields=['category_ids']
                    )
            )

        self.transform = module_transforms.Compose(
                self.transform,
                bbox_params=module_transforms.BboxParams(
                    format='pascal_voc',
                    label_fields=['category_ids']
                    )
            )

        self.csv[target_column] = self.csv[target_column].apply(eval)

    def __getitem__(self, idx: int):
        sample = self.get_raw(idx // self.expand_rate)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])
        
        output = {
            'input': sample['image'],
            'target_bboxes': torch.tensor(sample['bboxes']).type(torch.__dict__[self.target_dtype]),
            'target_labels': torch.tensor(sample['category_ids']).type(torch.__dict__[self.target_dtype]),
            'bbox_count': sample['bbox_count']
        }

        return output
        
    def get_raw(self, idx: int):
        record = self.csv.iloc[idx]
        image = self.read_image(record)
        row_annotations = record[self.target_column]

        bboxes = []
        labels = []
        for annotation in row_annotations:
            bbox = [annotation['x_min'], annotation['y_min'], annotation['x_max'], annotation['y_max']]
            label = annotation['label']
            bboxes.append(bbox)
            labels.append(label)

        bbox_count = len(bboxes)

        sample = {
            'image': image,
            'bboxes': bboxes,
            'category_ids': labels
            }

        if self.augment is not None:
            sample = self.augment(**sample)

        sample = self.transform(**sample)
        sample['bbox_count'] = bbox_count
        
        return sample

    @staticmethod
    def collate_fn(batch: dict) -> dict:
        """
        Add empty bbox and label into batch with different size of bboxes
        empty bbox = [0, 0, 0, 0]
        empty label = -1
        """
        # get sequence lengths
        max_length = 0
        for t in batch:
            max_length = max(max_length, t['bbox_count'])
       
        empty_box = [0]*4
        for t in batch:
            bbox_count = t['bbox_count']
            count_diff = max_length - bbox_count
            if count_diff != 0:
                append_bboxes = torch.tensor([empty_box for _ in range(count_diff)]).type(torch.long)
                append_label = torch.tensor([-1 for _ in range(count_diff)]).type(torch.long)
                if bbox_count != 0:
                    t['target_bboxes'] = torch.cat([t['target_bboxes'], append_bboxes])
                    t['target_labels'] = torch.cat([t['target_labels'], append_label])
                else:
                    t['target_bboxes'] = append_bboxes
                    t['target_labels'] = append_label
                
        batch = default_collate(batch)
        
        return batch