import torch
import unittest

from albumentations import HorizontalFlip, Resize, VerticalFlip
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from torchok.data.datasets.classification.classification import ImageClassificationDataset


class TestImageDataset:
    data_cls = None
    dataset_kwargs = {}
    ds_len = 0
    output_format = []

    def setUp(self) -> None:
        self.dataset_kwargs = dict(
            data_folder='tests/base_tests/data/datasets/data',
            transform=Compose([Resize(224, 224), ToTensorV2()], p=1.0),
        )
        self.augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def create_dataset(self):
        return self.data_cls(**self.dataset_kwargs)

    def test_len(self):
        ds = self.create_dataset()
        self.assertEqual(len(ds), self.ds_len)

    def test_shape_when_transformed(self):
        ds = self.create_dataset()
        self.assertTupleEqual(ds[0]['image'].shape, (3, 224, 224))

    def test_shape_when_grayscale(self):
        self.dataset_kwargs['grayscale'] = True
        ds = self.create_dataset()
        self.assertTupleEqual(ds[0]['image'].shape, (1, 224, 224))

    def test_augment_not_none(self):
        self.dataset_kwargs['augment'] = self.augment
        ds = self.create_dataset()
        self.assertTupleEqual(ds[0]['image'].shape, (3, 224, 224))

    def test_augment_not_none_and_grayscale(self):
        self.dataset_kwargs['augment'] = self.augment
        self.dataset_kwargs['grayscale'] = True
        ds = self.create_dataset()
        self.assertTupleEqual(ds[0]['image'].shape, (1, 224, 224))

    def test_input_dtype(self):
        self.dataset_kwargs['input_dtype'] = 'float16'
        ds = self.create_dataset()
        self.assertEqual(ds[0]['image'].dtype, torch.float16)

    def test_output_format(self):
        ds = self.create_dataset()
        self.assertListEqual(sorted(ds[0].keys()), sorted(self.output_format))


class TestClassificationDataset(TestImageDataset, unittest.TestCase):
    data_cls = ImageClassificationDataset

    def setUp(self) -> None:
        super().setUp()
        self.dataset_kwargs['data_folder'] = 'tests/base_tests/data/datasets/data'
        self.dataset_kwargs['csv_path'] = 'multiclass_test.csv'
        self.ds_len = 7
        self.output_format = ['image', 'target', 'index']

    def test_len(self):
        super().test_len()

    def test_shape_when_transformed(self):
        super().test_shape_when_transformed()

    def test_shape_when_grayscale(self):
        super().test_shape_when_grayscale()

    def test_augment_not_none(self):
        super().test_augment_not_none()

    def test_augment_not_none_and_grayscale(self):
        super().test_augment_not_none_and_grayscale()

    def test_input_dtype(self):
        super().test_input_dtype()

    def test_output_format(self):
        super().test_output_format()

    def test_target_multilabel_vector_len(self):
        self.dataset_kwargs['csv_path'] = 'multilabel_test.csv'
        self.dataset_kwargs['multilabel'] = True
        self.dataset_kwargs['num_classes'] = 2

        ds = self.create_dataset()
        self.assertEqual(len(ds[0]['target']), self.dataset_kwargs['num_classes'])
