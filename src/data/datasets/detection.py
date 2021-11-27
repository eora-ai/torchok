import cv2
import numpy as np
import pandas as pd
import torch
from typing import List
from src.data import transforms as module_transforms

from src.registry import DATASETS
from .classification import ImageDataset



@DATASETS.register_class
class ImageDetectionDataset(ImageDataset):
    def __init__(self, 
                target_cls_column='class_label',
                target_bbox_columns=['x_c', 'y_c', 'w', 'h'],
                input_bbox_format = 'yolo',
                output_bbox_format = 'pascal_voc',
                **dataset_params
     ):
        super().__init__(**dataset_params)
     
        # bbox format from Albumentations 
        # https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        assert input_bbox_format in ['pascal_voc', 'albumentations', 'coco', 'yolo'], \
            'Not correct bbox format'
        assert output_bbox_format in ['pascal_voc', 'albumentations', 'coco', 'yolo'], \
            'Not correct bbox format'

        if self.bbox_augment is not None:
            self.bbox_augment = module_transforms.Compose(
                self.bbox_augment,
                bbox_params=module_transforms.BboxParams(
                    format=input_bbox_format,
                    label_fields=['category_ids']
                    )
            )
     
        self.target_cls_column = target_cls_column
        self.target_bbox_columns = target_bbox_columns
        self.name2label = {'fawn': 0, 'reindeer': 1}
    

    def __getitem__(self, idx: int):
        sample = self.get_raw(idx // self.expand_rate)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])
        # if not self.test_mode:
        #     sample['class_labels'] = sample['class_labels'].type(torch.__dict__[self.target_dtype])
        
        output = {
            'input': sample['image'],
            'gt_bboxes': sample['bboxes'],
            'gt_labels': sample['category_ids']
        }
        return output
        
    def get_raw(self, idx: int):
        record = self.csv.iloc[idx]
        image = self.read_image(record)
        all_bboxes_df = self.csv.loc[self.csv['image_path'] == record.image_path]

        bboxes = []
        labels = []

        for i in all_bboxes_df.index:
            row = all_bboxes_df.iloc[i]
            bbox = [row[el_name] for el_name in self.target_bbox_columns] 
            label = self.name2label[row[self.target_cls_column]]
            bboxes.append(bbox)
            labels.append(label)
        
        image = self.augment(image=image)['image']


        sample = {
            'image': image,
            'bboxes': bboxes,
            'category_ids': labels
            }
        if self.bbox_augment is not None:
            sample = self.bbox_augment(**sample)

        sample['image'] = self.transform(image=sample['image'])['image']
        return sample

    
