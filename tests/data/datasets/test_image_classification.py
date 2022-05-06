import unittest
from pathlib import Path

import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip

from src.data.datasets.classification.image_classification import ImageClassificationDataset


class TestClassificationMulticlass(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.__root_dir = Path(__file__).parent
        self.__data_folder = self.__root_dir / 'data/'
        self.__csv_path = self.__root_dir / 'data/multiclass_test.csv'
        self.__num_classes = 3
        self.__transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self.__augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def test_len(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path,  self.__num_classes,
                                               self.__transform)
        self.assertEqual(len(self.__ds), 7)

    def test_shape_when_transformed(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform)
        self.assertTupleEqual(self.__ds[0]['image'].shape, (3, 224, 224))

    def test_when_grayscale(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, grayscale=True)
        self.assertTupleEqual(self.__ds[0]['image'].shape, (1, 224, 224))

    def test_input_dtype_when_specified(self):
        self.__input_dtype = 'float32'
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, image_dtype=self.__input_dtype)
        self.assertEqual(self.__ds[0]['image'].dtype, torch.__dict__[self.__input_dtype])

    def test_target_dtype_when_specified(self):
        self.__target_dtype = 'float32'
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, target_dtype=self.__target_dtype)
        self.assertEqual(self.__ds[0]['target'].dtype, torch.__dict__[self.__target_dtype])

    def test_input_when_test_mode_true(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, test_mode=True)
        self.assertListEqual([*self.__ds[0]], ['image', 'index'])

    def test_getitem_when_test_mode_false(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform)
        self.assertListEqual([*self.__ds[0]], ['image', 'index', 'target'])

    def test_when_lazy_init_true(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, multilabel=False, lazy_init=True)
        self.assertEqual(self.__ds[0]['target'].item(), 0)

    def test_when_augment_not_none(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, augment=self.__augment)
        self.assertTupleEqual(self.__ds[0]['image'].shape, (3, 224, 224))

    def test_when_augment_not_none_and_grayscale_true(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, augment=self.__augment, grayscale=True)
        self.assertTupleEqual(self.__ds[0]['image'].shape, (1, 224, 224))


class TestClassificationMultilabel(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.__root_dir = Path(__file__).parent
        self.__data_folder = self.__root_dir / 'data/'
        self.__csv_path = self.__root_dir / 'data/multilabel_test.csv'
        self.__num_classes = 2
        self.__transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)

    def test_len(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, multilabel=True)
        self.assertEqual(len(self.__ds), 4)

    def test_getitem_when_test_mode_true(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, test_mode=True, multilabel=True)
        self.assertListEqual([*self.__ds[0]], ['image', 'index'])

    def test_getitem_when_test_mode_false(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, test_mode=False, multilabel=True)
        self.assertListEqual([*self.__ds[0]], ['image', 'index', 'target'])

    def test_target_multihot_vector_len(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, multilabel=True)
        self.assertEqual(len(self.__ds[0]['target']), self.__num_classes)

    def test_when_lazy_init_true(self):
        self.__ds = ImageClassificationDataset(self.__data_folder, self.__csv_path, self.__num_classes,
                                               self.__transform, multilabel=True, lazy_init=True)
        self.assertEqual(len(self.__ds[0]['target']), self.__num_classes)


if __name__ == "__main__":
    unittest.main()
