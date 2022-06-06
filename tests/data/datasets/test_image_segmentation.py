import unittest

from tests.data.datasets.test_image_classification import TestImageDataset
from src.data.datasets.segmentation.image_segmentation import ImageSegmentationDataset


class TestSegmentationDataset(TestImageDataset, unittest.TestCase):
    def __init__(self, method_name=...) -> None:
        kwargs = {}
        args = []
        super().__init__(ImageSegmentationDataset, 'data/segmentation_data/',
                         'segmentation_test.csv', args, kwargs, method_name)

    def test_len(self):
        super().test_len(num_samples=3)

    def test_shape_when_transformed(self):
        super().test_shape_when_transformed(image_shape=(3, 224, 224))

    def test_shape_when_grayscale_true(self):
        super().test_shape_when_grayscale_true(image_shape=(1, 224, 224))

    def test_when_augment_not_none(self):
        super().test_when_augment_not_none(image_shape=(3, 224, 224))

    def test_when_augment_not_none_and_grayscale_true(self):
        super().test_when_augment_not_none_and_grayscale_true(image_shape=(1, 224, 224))


if __name__ == "__main__":
    unittest.main()
