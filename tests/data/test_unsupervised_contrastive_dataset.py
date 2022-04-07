import os
import unittest

import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip

from src.data.datasets.representation.unsupervised_contrastive_dataset import UnsupervisedContrastiveDataset


class TestUnsupervisedContrastiveDataset(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.data_folder = os.getcwd() + '/tests/datasets/data/'
        self.csv_path = os.getcwd() + '/tests/datasets/data/'\
                                      'multiclass_test.csv'
        self.transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self.augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def test_len(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.csv_path, self.transform)
        self.assertEqual(len(self.ds), 5)

    def test_shape_when_transformed(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.csv_path, self.transform)
        self.assertEqual(list(self.ds[0]['input_0'].shape), [3, 224, 224])

    def test_when_grayscale(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.csv_path, self.transform,
                                                 grayscale=True)
        self.assertEqual(list(self.ds[0]['input_0'].shape), [1, 224, 224])

    def test_input_dtype_when_specified(self):
        input_dtype = 'float32'
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.csv_path, self.transform,
                                                 input_dtype=input_dtype)
        self.assertEqual(self.ds[0]['input_0'].dtype, torch.__dict__[input_dtype])

    def test_output_format(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.csv_path, self.transform)
        self.assertListEqual([*self.ds[0]], ['input_0', 'input_1', 'index'])

    def test_when_augment_not_none(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.csv_path, self.transform,
                                                 augment=self.augment)
        self.assertTupleEqual(self.ds[0]['input_0'].shape, (3, 224, 224))
        self.assertTupleEqual(self.ds[0]['input_1'].shape, (3, 224, 224))

    def test_when_augment_not_none_and_grayscale_true(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.csv_path, self.transform,
                                                 augment=self.augment, grayscale=True)
        self.assertTupleEqual(self.ds[0]['input_0'].shape, (1, 224, 224))
        self.assertTupleEqual(self.ds[0]['input_1'].shape, (1, 224, 224))


if __name__ == "__main__":
    unittest.main()
