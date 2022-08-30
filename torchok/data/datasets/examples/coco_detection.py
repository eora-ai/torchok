import json
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchvision.datasets.utils import download_and_extract_archive

from torchok.constructor import DATASETS
from torchok.data.datasets.detection import DetectionDataset


@DATASETS.register_class
class COCODetection(DetectionDataset):
    """A class represent detection COCO dataset https://cocodataset.org/#home.

    The COCO Object Detection Task is designed to push the state of the art in object detection forward.
    COCO features two object detection tasks: using either bounding box output or object segmentation output
    (the latter is also known as instance segmentation). 
    """
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

    train_csv = 'train.csv'
    valid_csv = 'valid.csv'

    def __init__(self,
                 train: bool,
                 download: bool,
                 data_folder: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 target_dtype: str = 'long',
                 bbox_dtype: str = 'float32',
                 grayscale: bool = False,
                 test_mode: bool = False,
                 min_area: float = 0.0,
                 min_visibility: float = 0.0,
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
            target_dtype: Data type of the torch tensors related to the bboxes labels.
            bbox_dtype: Data type of the torch tensors related to the bboxes.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
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
        """
        self.data_folder = Path(data_folder)
        self.path = self.data_folder / self.base_folder

        if download:
            self._download()

            # Create train csv
            train_annotation_path = self.path / 'annotations/instances_train2017.json'
            train_csv_path = self.path / self.train_csv
            train_image_folder = Path('train2017')
            self.create_csv(train_annotation_path, train_image_folder, train_csv_path)

            # Create valid csv
            val_annotation_path = self.path / 'annotations/instances_val2017.json'
            val_csv_path = self.path / self.valid_csv
            val_image_folder = Path('val2017')
            self.create_csv(val_annotation_path, val_image_folder, val_csv_path)

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
            bbox_dtype=bbox_dtype,
            grayscale=grayscale,
            test_mode=test_mode,
            min_area=min_area,
            min_visibility=min_visibility
        )

    def create_csv(self, json_path: str, image_folder: str, output_name: str):
        """Create train-valid csv for loaded COCO dataset.

        Args:
            json_path: COCO json annotation file path.
            image_folder: COCO images folder.
            output_name: Csv save name.
        """
        # Read json
        f = open(json_path)
        data = json.load(f)

        # Create annotations - id, bboxes and labels
        annot_df = pd.DataFrame()
        bboxes = []
        labels = []
        ids = []
        for ann in data['annotations']:
            ids.append(ann['image_id'])
            labels.append(ann['category_id'])
            bboxes.append(ann['bbox'])

        annot_df['id'] = ids
        annot_df['bbox'] = bboxes
        annot_df['label']  = labels

        # Create image_path - id dataframe
        df = pd.DataFrame()
        image_paths = []
        ids = []

        for ann in data['images']:
            image_paths.append((image_folder / ann['file_name']).as_posix())
            ids.append(ann['id'])

        df['image_path'] = image_paths
        df['id'] = ids

        # Concatenate image_path with bboxes and labels by it's id
        all_bboxes = []
        all_labels = []
        for i in range(len(df)):
            curr_id = df.iloc[i]['id']
            curr_df = annot_df.loc[annot_df['id'] == curr_id]
            bboxes = curr_df.bbox.tolist()
            labels = curr_df.label.tolist()
            all_bboxes.append(str(bboxes))
            all_labels.append(str(labels))

        df['bbox'] = all_bboxes
        df['label'] = all_labels

        # Drop id column
        df = df.drop(columns=['id'])
        
        # Save csv
        df.to_csv(output_name, index=False)

    def _download(self) -> None:
        """Download archive by url to specific folder."""
        if self.path.is_dir():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.train_data_url, self.path.as_posix(),
                                         filename=self.train_data_filename, md5=self.train_data_hash)
            download_and_extract_archive(self.valid_data_url, self.path.as_posix(),
                                         filename=self.valid_data_filename, md5=self.valid_data_hash)
            download_and_extract_archive(self.annotations_url, self.path.as_posix(),
                                         filename=self.annotations_filename, md5=self.annotations_hash)
