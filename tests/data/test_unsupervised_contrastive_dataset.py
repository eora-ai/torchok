import unittest

import torch
from albumentations import Resize
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from src.data.datasets.representation.unsupervised_contrastive_dataset import UnsupervisedContrastiveDataset


class TestUnsupervisedContrastiveDataset(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.data_folder = '/workdir/'
        self.path_to_datalist = '/workdir/vpatrushev/torchOK2/torchok/tests/datasets/classification/'\
                                'test_classification.csv'
        self.transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)

    def test_len_default(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.path_to_datalist, self.transform)
        self.assertEqual(len(self.ds), 3180)

    def test_shape_default(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.path_to_datalist, self.transform)
        self.assertEqual(list(self.ds[0]['input_0'].shape), [3, 224, 224])

    def test_grayscale(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.path_to_datalist, self.transform,
                                                 grayscale=True)
        self.assertEqual(list(self.ds[0]['input_0'].shape), [1, 224, 224])

    def test_input_dtype(self):
        input_dtype = 'float32'
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.path_to_datalist, self.transform,
                                                 input_dtype=input_dtype)
        self.assertEqual(self.ds[0]['input_0'].dtype, torch.__dict__[input_dtype])

    def test_output_format(self):
        self.ds = UnsupervisedContrastiveDataset(self.data_folder, self.path_to_datalist, self.transform)
        self.assertListEqual([*self.ds[0]], ['input_0', 'input_1', 'index'])


if __name__ == "__main__":
    unittest.main()
