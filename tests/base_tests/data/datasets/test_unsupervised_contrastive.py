import unittest

import torch

from tests.base_tests.data.datasets.test_image_classification import TestImageDataset
from torchok.data.datasets.representation.unsupervised_contrastive_dataset import UnsupervisedContrastiveDataset


class TestUnsupervisedContrastiveDataset(TestImageDataset, unittest.TestCase):
    data_cls = UnsupervisedContrastiveDataset

    def setUp(self) -> None:
        super().setUp()
        self.dataset_kwargs['data_folder'] = 'tests/base_tests/data/datasets/data/'
        self.dataset_kwargs['csv_path'] = 'unsupervised_contrastive_test.csv'
        self.ds_len = 7
        self.output_format = ['image_0', 'image_1', 'index']

    def test_len(self):
        super().test_len()

    def test_shape_when_transformed(self):
        ds = self.create_dataset()
        self.assertTupleEqual(ds[0]['image_0'].shape, (3, 224, 224))

    def test_shape_when_grayscale(self):
        self.dataset_kwargs['image_format'] = 'gray'
        ds = self.create_dataset()
        self.assertTupleEqual(ds[0]['image_0'].shape, (1, 224, 224))

    def test_augment_not_none(self):
        self.dataset_kwargs['augment'] = self.augment
        ds = self.create_dataset()
        self.assertTupleEqual(ds[0]['image_0'].shape, (3, 224, 224))
        self.assertTupleEqual(ds[0]['image_1'].shape, (3, 224, 224))

    def test_augment_not_none_and_grayscale(self):
        self.dataset_kwargs['augment'] = self.augment
        self.dataset_kwargs['image_format'] = 'gray'
        ds = self.create_dataset()
        self.assertTupleEqual(ds[0]['image_0'].shape, (1, 224, 224))
        self.assertTupleEqual(ds[0]['image_1'].shape, (1, 224, 224))

    def test_input_dtype(self):
        self.dataset_kwargs['input_dtype'] = 'float16'
        ds = self.create_dataset()
        self.assertEqual(ds[0]['image_0'].dtype, torch.float16)

    def test_output_format(self):
        super().test_output_format()
