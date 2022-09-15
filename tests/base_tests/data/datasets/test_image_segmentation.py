import unittest

from tests.base_tests.data.datasets.test_image_classification import TestImageDataset
from torchok.data.datasets.segmentation.image_segmentation import ImageSegmentationDataset


class TestSegmentationDataset(TestImageDataset, unittest.TestCase):
    data_cls = ImageSegmentationDataset

    def setUp(self) -> None:
        super().setUp()
        self.dataset_kwargs['data_folder'] = 'tests/base_tests/data/datasets/data/segmentation_data'
        self.dataset_kwargs['csv_path'] = 'segmentation_test.csv'
        self.dataset_kwargs['target_column'] = 'mask'
        self.ds_len = 3
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

    def test_input_target_spatial_shape_equality(self):
        ds = self.create_dataset()
        sample = ds[0]
        self.assertTupleEqual(sample['image'].shape[-2:], sample['target'].shape[-2:])
