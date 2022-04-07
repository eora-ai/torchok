import os
import unittest

import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip

from src.data.datasets.classification.image_classification import ImageClassificationDataset


class TestClassificationMulticlass(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.data_folder = os.getcwd() + '/tests/datasets/data/'
        self.csv_path = os.getcwd() + '/tests/datasets/data/'\
                                      'multiclass_test.csv'
        self.num_classes = 3
        self.transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self.augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def test_len(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path,  self.num_classes,
                                             self.transform)
        self.assertEqual(len(self.ds), 5)

    def test_shape_when_transformed(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform)
        self.assertTupleEqual(self.ds[0]['input'].shape, (3, 224, 224))

    def test_when_grayscale(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, grayscale=True)
        self.assertEqual(list(self.ds[0]['input'].shape), [1, 224, 224])

    def test_input_dtype_when_specified(self):
        input_dtype = 'float32'
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, input_dtype=input_dtype)
        self.assertEqual(self.ds[0]['input'].dtype, torch.__dict__[input_dtype])

    def test_target_dtype_when_specified(self):
        target_dtype = 'float32'
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, target_dtype=target_dtype)
        self.assertEqual(self.ds[0]['target'].dtype, torch.__dict__[target_dtype])

    def test_input_when_test_mode_true(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, test_mode=True)
        self.assertListEqual([*self.ds[0]], ['input', 'index'])

    def test_getitem_when_test_mode_false(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform)
        self.assertListEqual([*self.ds[0]], ['input', 'index', 'target'])

    def test_when_lazy_init_true(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, multilabel=False, lazy_init=True)
        self.assertEqual(self.ds[0]['target'].item(), 0)

    def test_when_augment_not_none(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, augment=self.augment)
        self.assertTupleEqual(self.ds[0]['input'].shape, (3, 224, 224))

    def test_when_augment_not_none_and_grayscale_true(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, augment=self.augment, grayscale=True)
        self.assertTupleEqual(self.ds[0]['input'].shape, (1, 224, 224))


class TestClassificationMultilabel(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.data_folder = os.getcwd() + '/tests/datasets/data/'
        self.csv_path = os.getcwd() + '/tests/datasets/data/'\
                                      'multilabel_test.csv'
        self.num_classes = 2
        self.transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)

    def test_len(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, multilabel=True)
        self.assertEqual(len(self.ds), 4)

    def test_getitem_when_test_mode_true(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, test_mode=True, multilabel=True)
        self.assertListEqual([*self.ds[0]], ['input', 'index'])

    def test_getitem_when_test_mode_false(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, test_mode=False, multilabel=True)
        self.assertListEqual([*self.ds[0]], ['input', 'index', 'target'])

    def test_target_multihot_vector_len(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, multilabel=True)
        self.assertEqual(len(self.ds[0]['target']), self.num_classes)

    def test_when_lazy_init_true(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.csv_path, self.num_classes,
                                             self.transform, multilabel=True, lazy_init=True)
        self.assertEqual(len(self.ds[0]['target']), self.num_classes)


if __name__ == "__main__":
    unittest.main()
