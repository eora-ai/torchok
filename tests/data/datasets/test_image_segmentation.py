import unittest
from pathlib import Path

import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip

from src.data.datasets.segmentation.image_segmentation import ImageSegmentationDataset


class TestClassificationMulticlass(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.__root_dir = Path(__file__).parent
        self.__data_folder = self.__root_dir / 'data/segmentation_data/'
        self.__csv_path = self.__root_dir / 'data/segmentation_data/segmentation_test.csv'
        self.__transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self.__augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def test_len(self):
        self.__ds = ImageSegmentationDataset(self.__data_folder, self.__csv_path,
                                             self.__transform)
        self.assertEqual(len(self.__ds), 3)

    def test_shape_when_transformed(self):
        self.__ds = ImageSegmentationDataset(self.__data_folder, self.__csv_path,
                                             self.__transform)
        self.assertTupleEqual(self.__ds[0]['image'].shape, (3, 224, 224))

    def test_target_shape_when_transformed(self):
        self.__ds = ImageSegmentationDataset(self.__data_folder, self.__csv_path,
                                             self.__transform)
        self.assertTupleEqual(self.__ds[0]['target'].shape, (224, 224))

    def test_input_dtype_when_specified(self):
        self.__input_dtype = 'float32'
        self.__ds = ImageSegmentationDataset(self.__data_folder, self.__csv_path,
                                             self.__transform, image_dtype=self.__input_dtype)
        self.assertEqual(self.__ds[0]['image'].dtype, torch.__dict__[self.__input_dtype])

    def test_input_when_test_mode_true(self):
        self.__ds = ImageSegmentationDataset(self.__data_folder, self.__csv_path,
                                             self.__transform, test_mode=True)
        self.assertListEqual([*self.__ds[0]], ['image', 'index'])

    def test_getitem_when_test_mode_false(self):
        self.__ds = ImageSegmentationDataset(self.__data_folder, self.__csv_path,
                                             self.__transform)
        self.assertListEqual([*self.__ds[0]], ['image', 'index', 'target'])

    def test_when_augment_not_none(self):
        self.__ds = ImageSegmentationDataset(self.__data_folder, self.__csv_path,
                                             self.__transform, augment=self.__augment)
        self.assertTupleEqual(self.__ds[0]['image'].shape, (3, 224, 224))


if __name__ == "__main__":
    unittest.main()
