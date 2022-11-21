import cv2
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from pathlib import Path
from typing import Optional, Union

from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchvision.datasets.utils import download_and_extract_archive

from torchok.constructor import DATASETS
from torchok.data.datasets.segmentation.image_segmentation import ImageSegmentationDataset


@DATASETS.register_class
class COCOSegmentation(ImageSegmentationDataset):
    """A class represent detection COCO dataset https://cocodataset.org/#home.

    The COCO Object Detection Task is designed to push the state of the art in object detection forward.
    COCO features two object detection tasks: using either bounding box output or object segmentation output
    (the latter is also known as instance segmentation).

    COCO dataset has 81 categories where 0 - background label. Train set contains 118287 images, validation set - 5000.

    This Dataset occupies 21 Gb of memory.
    """
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    label_mapping = {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15,
        17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28,
        33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41,
        47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54,
        60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67,
        77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80
    }

    base_folder = 'COCO'

    train_data_filename = 'train2017.zip'
    train_data_url = 'http://images.cocodataset.org/zips/train2017.zip'
    train_data_hash = 'cced6f7f71b7629ddf16f17bbcfab6b2'

    valid_data_filename = 'valid2017.zip'
    valid_data_url = 'http://images.cocodataset.org/zips/val2017.zip'
    valid_data_hash = '442b8da7639aecaf257c1dceb8ba8c80'

    annotations_filename = 'annotations.zip'
    annotations_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    annotations_hash = 'f4bbac642086de4f52a3fdda2de5fa2c'

    train_csv = 'train_segmentation.csv'
    valid_csv = 'valid_segmentation.csv'

    def __init__(self,
                 train: bool,
                 download: bool,
                 data_folder: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 target_dtype: str = 'long',
                 channel_order: str = 'rgb',
                 grayscale: bool = False,
                 test_mode: bool = False,
                 ):
        """Init SweetPepper.

        Args:
            train: If True, train dataset will be used, else - test dataset.
            download: If True, data will be downloaded and save to data_folder.
            data_folder: Directory with all the images.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of the torch tensors related to the image.
            target_dtype: Data type of the torch tensors related to the target mask.
            channel_order: Order of channel, candidates are `bgr` and `rgb`.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
        """
        self.data_folder = Path(data_folder)
        self.path = self.data_folder / self.base_folder

        if download:
            self._download()

        # Create train csv
        train_df_path = self.path / self.train_csv
        if not train_df_path.exists():
            train_annotation_path = self.path / 'annotations/instances_train2017.json'
            train_mask_save_folder = self.path / 'train2017'
            self.create_annotation(train_annotation_path, train_mask_save_folder, train_df_path)

        # Create valid csv
        val_df_path = self.path / self.valid_csv
        if not val_df_path.exists():
            val_annotation_path = self.path / 'annotations/instances_val2017.json'
            val_mask_save_folder = self.path / 'val2017'
            self.create_annotation(val_annotation_path, val_mask_save_folder, val_df_path)

        if not self.path.is_dir():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        csv_path = self.train_csv if train else self.valid_csv

        super().__init__(
            data_folder=self.path,
            csv_path=csv_path,
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            target_dtype=target_dtype,
            channel_order=channel_order,
            grayscale=grayscale,
            test_mode=test_mode,
        )

    def create_annotation(self, json_path: str, mask_folder: str, save_df_path: str):
        """Create train-valid csv for loaded COCO dataset.

        Args:
            json_path: COCO json annotation file path.
            mask_folder: COCO mask save folder.
            save_df_path: Pickle save name.
        """
        image_paths = []
        mask_paths = []
        coco = COCO(json_path)
        ids = coco.getImgIds()
        for image_id in ids:
            # add image_path
            img_params = coco.loadImgs(image_id)[0]
            height = img_params['height']
            width = img_params['width']
            file_name = img_params['file_name']
            image_paths.append((mask_folder / file_name).as_posix())
            # create mask
            annIds = coco.getAnnIds(image_id)
            anns = coco.loadAnns(annIds)
            mask = np.zeros((height, width), dtype=np.int32)
            for ann in anns:
                mask += self.label_mapping[ann['category_id']] * coco.annToMask(ann)
            # save mask in same folder
            mask_file_name = 'mask_' + file_name.split('.')[0] + '.png'
            mask_path = (mask_folder / mask_file_name).as_posix()
            mask_paths.append(mask_path)
            cv2.imwrite(mask_path, mask)

        df = pd.DataFrame()
        df['image_path'] = image_paths
        df['mask_path'] = mask_paths

        # Save csv
        df.to_csv(save_df_path, index=False)

    def _download(self) -> None:
        """Download archive by url to specific folder."""
        if self.path.is_dir():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.train_data_url, self.path.as_posix(), filename=self.train_data_filename,
                                         md5=self.train_data_hash, remove_finished=True)
            download_and_extract_archive(self.valid_data_url, self.path.as_posix(), filename=self.valid_data_filename,
                                         md5=self.valid_data_hash, remove_finished=True)
            download_and_extract_archive(self.annotations_url, self.path.as_posix(), filename=self.annotations_filename,
                                         md5=self.annotations_hash, remove_finished=True)
