import unittest
from pathlib import Path

import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip

from torchok.data.datasets.representation.unsupervised_contrastive_dataset import UnsupervisedContrastiveDataset


class TestUnsupervisedContrastiveDataset(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.__root_dir = Path(__file__).parent
        self.__data_folder = self.__root_dir / 'data/'
        self.__csv_path = self.__root_dir / 'data/unsupervised_contrastive_test.csv'
        self.__transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self.__augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def test_len(self):
        self.__ds = UnsupervisedContrastiveDataset(self.__data_folder, self.__csv_path, self.__transform)
        self.assertEqual(len(self.__ds), 7)

    def test_shape_when_transformed(self):
        self.__ds = UnsupervisedContrastiveDataset(self.__data_folder, self.__csv_path, self.__transform)
        self.assertTupleEqual(self.__ds[0]['image_0'].shape, (3, 224, 224))

    def test_when_grayscale(self):
        self.__ds = UnsupervisedContrastiveDataset(self.__data_folder, self.__csv_path, self.__transform,
                                                   grayscale=True)
        self.assertTupleEqual(self.__ds[0]['image_0'].shape, (1, 224, 224))

    def test_input_dtype_when_specified(self):
        input_dtype = 'float32'
        self.__ds = UnsupervisedContrastiveDataset(self.__data_folder, self.__csv_path, self.__transform,
                                                   image_dtype=input_dtype)
        self.assertEqual(self.__ds[0]['image_0'].dtype, torch.__dict__[input_dtype])

    def test_output_format(self):
        self.__ds = UnsupervisedContrastiveDataset(self.__data_folder, self.__csv_path, self.__transform)
        self.assertListEqual([*self.__ds[0]], ['image_0', 'image_1', 'index'])

    def test_when_augment_not_none(self):
        self.__ds = UnsupervisedContrastiveDataset(self.__data_folder, self.__csv_path, self.__transform,
                                                   augment=self.__augment)
        self.assertTupleEqual(self.__ds[0]['image_0'].shape, (3, 224, 224))
        self.assertTupleEqual(self.__ds[0]['image_1'].shape, (3, 224, 224))

    def test_when_augment_not_none_and_grayscale_true(self):
        self.__ds = UnsupervisedContrastiveDataset(self.__data_folder, self.__csv_path, self.__transform,
                                                   augment=self.__augment, grayscale=True)
        self.assertTupleEqual(self.__ds[0]['image_0'].shape, (1, 224, 224))
        self.assertTupleEqual(self.__ds[0]['image_1'].shape, (1, 224, 224))


if __name__ == "__main__":
    unittest.main()
