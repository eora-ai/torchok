import unittest
from pathlib import Path

import numpy as np

from tests.data.datasets.test_image_classification import TestImageDataset
from torchok.data.datasets.detection import DetectionDataset 


class TestDetectionDataset(TestImageDataset, unittest.TestCase):
    data_cls = DetectionDataset

    def setUp(self) -> None:
        super().setUp()
        root = Path('tests/data/datasets/detection')
        self.dataset_kwargs['data_folder'] = root
        self.dataset_kwargs['annotation_path'] = 'coco_valid.pkl'
        self.ds_len = 5
        self.output_format = ['image', 'index', 'label', 'bboxes']

    def test_len(self):
        super().test_len()

    def test_shape_when_transformed(self):
        super().test_shape_when_transformed()

    def test_shape_when_grayscale(self):
        super().test_shape_when_grayscale()

    def test_input_dtype(self):
        super().test_input_dtype()

    def test_output_format(self):
        ds = self.create_dataset()
        self.assertListEqual(list(ds[0].keys()), self.output_format)

    def test_output_format_when_annotation_is_csv(self):
        self.dataset_kwargs['annotation_path'] = 'coco_valid.csv'
        ds = self.create_dataset()
        self.assertListEqual(list(ds[0].keys()), self.output_format)

    def test_bboxes_when_transformed(self):
        # Convert 448x448 to 224x224 in coco format
        img = np.zeros((448, 448, 3))
        bboxes = [[0, 0, 448, 448], [0, 0, 224, 224]]
        output_bboxes = [[0, 0, 224, 224], [0, 0, 112, 112]]
        label = [0, 1]

        sample = {'image': img, 'bboxes': bboxes, 'label': label}

        transform = self.create_dataset().transform
        transformed_sample = transform(**sample)

        # need to convert to int
        transformed_bboxes = np.array(transformed_sample['bboxes'], dtype=np.long).tolist()

        self.assertListEqual(output_bboxes, transformed_bboxes)
