from typing import List, Tuple
import unittest
from pathlib import Path

import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip

from src.data.datasets.classification.image_classification import ImageClassificationDataset


class TestImageDataset:

    def __init__(self, data_cls, data_folder, csv_path, args, kwargs, method_name) -> None:
        super().__init__(method_name)
        self.__root_dir = Path(__file__).parent
        self._data_folder = self.__root_dir / data_folder
        self._csv_path = self.__root_dir / data_folder / csv_path
        self._transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self._augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])
        self._dataset_args = args
        self._dataset_kwargs = kwargs
        self._data_cls = data_cls

    def test_len(self, num_samples):
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, **self._dataset_kwargs)
        self.assertEqual(len(self.__ds), num_samples)

    def test_shape_when_transformed(self, image_shape: Tuple[int, int, int]):
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, **self._dataset_kwargs)
        self.assertTupleEqual(self.__ds[0]['image'].shape, image_shape)

    def test_shape_when_grayscale_true(self, image_shape: Tuple[int, int, int]):
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, grayscale=True, **self._dataset_kwargs)
        self.assertTupleEqual(self.__ds[0]['image'].shape, image_shape)

    def test_input_dtype_when_specified(self, image_dtype: str = 'float32'):
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, image_dtype=image_dtype, **self._dataset_kwargs)
        self.assertEqual(self.__ds[0]['image'].dtype, torch.__dict__[image_dtype])

    def test_target_dtype_when_specified(self, target_dtype: str = 'float32'):
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, target_dtype=target_dtype, **self._dataset_kwargs)
        self.assertEqual(self.__ds[0]['target'].dtype, torch.__dict__[target_dtype])

    def test_input_when_test_mode_true(self, sample_keys: List[str] = None):
        sample_keys = ['image', 'index'] if sample_keys is None else sample_keys
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, test_mode=True, **self._dataset_kwargs)
        self.assertListEqual([*self.__ds[0]], sample_keys)

    def test_getitem_when_test_mode_false(self, sample_keys: List[str] = None):
        sample_keys = ['image', 'index', 'target'] if sample_keys is None else sample_keys
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, test_mode=False, **self._dataset_kwargs)
        self.assertListEqual([*self.__ds[0]], sample_keys)

    def test_when_augment_not_none(self, image_shape: Tuple[int, int, int]):
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, augment=self._augment, **self._dataset_kwargs)
        self.assertTupleEqual(self.__ds[0]['image'].shape, image_shape)

    def test_when_augment_not_none_and_grayscale_true(self, image_shape: Tuple[int, int, int]):
        self.__ds = self._data_cls(self._data_folder, self._csv_path, *self._dataset_args,
                                   self._transform, augment=self._augment, grayscale=True, **self._dataset_kwargs)
        self.assertTupleEqual(self.__ds[0]['image'].shape, image_shape)


class TestClassificationMulticlass(TestImageDataset, unittest.TestCase):
    def __init__(self, method_name=...) -> None:
        self.__num_classes = 3
        kwargs = {}
        args = [self.__num_classes]
        super().__init__(ImageClassificationDataset, 'data/', 'multiclass_test.csv', args, kwargs, method_name)

    def test_len(self):
        super().test_len(num_samples=7)

    def test_shape_when_transformed(self):
        super().test_shape_when_transformed(image_shape=(3, 224, 224))

    def test_shape_when_grayscale_true(self):
        super().test_shape_when_grayscale_true(image_shape=(1, 224, 224))

    def test_when_augment_not_none(self):
        super().test_when_augment_not_none(image_shape=(3, 224, 224))

    def test_when_augment_not_none_and_grayscale_true(self):
        super().test_when_augment_not_none_and_grayscale_true(image_shape=(1, 224, 224))


class TestClassificationMultilabel(TestImageDataset, unittest.TestCase):
    def __init__(self, methodName=...) -> None:
        self.__num_classes = 2
        kwargs = {'multilabel': True}
        args = [self.__num_classes]
        super().__init__(ImageClassificationDataset, 'data/', 'multilabel_test.csv',
                         args, kwargs, methodName)

    def test_len(self):
        super().test_len(num_samples=4)

    def test_shape_when_transformed(self):
        super().test_shape_when_transformed(image_shape=(3, 224, 224))

    def test_shape_when_grayscale_true(self):
        super().test_shape_when_grayscale_true(image_shape=(1, 224, 224))

    def test_when_augment_not_none(self):
        super().test_when_augment_not_none(image_shape=(3, 224, 224))

    def test_when_augment_not_none_and_grayscale_true(self):
        super().test_when_augment_not_none_and_grayscale_true(image_shape=(1, 224, 224))

    def test_target_multihot_vector_len(self):
        self.__ds = ImageClassificationDataset(self._data_folder, self._csv_path, self.__num_classes,
                                               self._transform, multilabel=True)
        self.assertEqual(len(self.__ds[0]['target']), self.__num_classes)


if __name__ == "__main__":
    unittest.main()
