import unittest

import torch
from albumentations import Resize
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2

from src.data.datasets.classification.image_classification import ImageClassificationDataset


class TestClassification(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.data_folder = '/workdir/'
        self.path_to_datalist = 'test_classification.csv'
        self.transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)

    def test_len_default(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform)
        self.assertEqual(len(self.ds), 3180)

    def test_expand_rate(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform, expand_rate=2)
        self.assertEqual(len(self.ds), 6360)

    def test_shape_default(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform)
        self.assertEqual(list(self.ds[0]['input'].shape), [3, 224, 224])

    def test_grayscale(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform, grayscale=True)
        self.assertEqual(list(self.ds[0]['input'].shape), [1, 224, 224])

    def test_input_dtype(self):
        input_dtype = 'float32'
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform,
                                             input_dtype=input_dtype)
        self.assertEqual(self.ds[1]['input'].dtype, torch.__dict__[input_dtype])

    def test_input_test_mode_true(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform, test_mode=True)
        self.assertListEqual([*self.ds[0]], ['input', 'index'])

    def test_input_test_mode_false(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform)
        self.assertListEqual([*self.ds[0]], ['input', 'index', 'target'])

    def test_multilabel_true(self):
        num_classes = 9
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform, multilabel=True,
                                             num_classes=num_classes)
        self.assertEqual(len(self.ds[0]['target']), num_classes)

    def test_lazy_init_multilabel(self):
        num_classes = 9
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform, multilabel=True,
                                             num_classes=9, lazy_init_multilabel=True)
        self.assertEqual(len(self.ds[0]['target']), num_classes)


class TestClassificationMultilabel(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.data_folder = '/workdir/'
        self.path_to_datalist = 'test_classification_multilabel.csv'
        self.transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)

    def test_len_default(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform)
        self.assertEqual(len(self.ds), 3180)

    def test_expand_rate(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform, expand_rate=2)
        self.assertEqual(len(self.ds), 6360)

    def test_input_test_mode_true(self):
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform, test_mode=True)
        self.assertListEqual([*self.ds[0]], ['input', 'index'])

    def test_multilabel_true(self):
        num_classes = 9
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform,
                                             multilabel=True, num_classes=num_classes)
        self.assertEqual(len(self.ds[0]['target']), num_classes)

    def test_lazy_init_multilabel(self):
        num_classes = 9
        self.ds = ImageClassificationDataset(self.data_folder, self.path_to_datalist, self.transform, multilabel=True,
                                             num_classes=9, lazy_init_multilabel=True)
        self.assertEqual(len(self.ds[0]['target']), num_classes)


if __name__ == "__main__":
    unittest.main()
