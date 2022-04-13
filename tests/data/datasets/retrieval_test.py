import unittest
from pathlib import Path

import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip

from src.data.datasets.representation.each_to_each_retrieval_dataset import EachToEachRetrievalDataset
from src.data.datasets.representation.db_retrieval_dataset import DbRetrievalDataset


class TestEachToEachRetrievalDataset(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.__root_dir = Path(__file__).parent
        self.__data_folder = self.__root_dir / 'data/retrieval/orig/'
        self.__img_paths_csv_path = self.__root_dir / 'data/retrieval/toilet_paths.csv'
        self.__matches_csv_path = self.__root_dir / 'data/retrieval/toilet_match.csv'
        self.__transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self.__augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def test_len(self):
        self.__ds = EachToEachRetrievalDataset(self.__data_folder,
                                               self.__matches_csv_path,
                                               self.__img_paths_csv_path,
                                               self.__transform,
                                               self.__augment)
        self.assertEqual(len(self.__ds), 492)

    def test_shape_when_transformed(self):
        self.__ds = EachToEachRetrievalDataset(self.__data_folder,
                                               self.__matches_csv_path,
                                               self.__img_paths_csv_path,
                                               self.__transform,
                                               self.__augment)
        self.assertEqual(self.__ds[0]['image'].shape, (3, 224, 224))

    def test_when_grayscale(self):
        self.__ds = EachToEachRetrievalDataset(self.__data_folder,
                                               self.__matches_csv_path,
                                               self.__img_paths_csv_path,
                                               self.__transform,
                                               self.__augment,
                                               grayscale=True)

        self.assertEqual(self.__ds[0]['image'].shape, (1, 224, 224))

    def test_input_dtype_when_specified(self):
        self.__input_dtype = 'float32'
        self.__ds = EachToEachRetrievalDataset(self.__data_folder,
                                               self.__matches_csv_path,
                                               self.__img_paths_csv_path,
                                               self.__transform,
                                               self.__augment,
                                               input_dtype=self.__input_dtype)
        self.assertEqual(self.__ds[0]['image'].dtype, torch.__dict__[self.__input_dtype])

    def test_output_format(self):
        self.__ds = EachToEachRetrievalDataset(self.__data_folder,
                                               self.__matches_csv_path,
                                               self.__img_paths_csv_path,
                                               self.__transform,
                                               self.__augment)

        self.assertListEqual([*self.__ds[0]], ['image', 'index'])


class TestDbRetrievalDataset(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.__root_dir = Path(__file__).parent
        self.__db_folder = self.__root_dir / 'data/retrieval/orig/'
        self.__data_folder = self.__root_dir / 'data/retrieval/orig/'
        self.__include_path = self.__root_dir / 'data/retrieval/toilet_include.csv'
        self.__img_paths_csv_path = self.__root_dir / 'data/retrieval/toilet_paths.csv'
        self.__matches_csv_path = self.__root_dir / 'data/retrieval/toilet_match.csv'
        self.__transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self.__augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def test_len(self):
        self.__ds = DbRetrievalDataset(self.__data_folder,
                                       self.__matches_csv_path,
                                       self.__db_folder,
                                       self.__include_path,
                                       self.__img_paths_csv_path,
                                       self.__transform,
                                       self.__augment)
        self.assertEqual(len(self.__ds), 820)

    def test_shape_when_transformed(self):
        self.__ds = DbRetrievalDataset(self.__data_folder,
                                       self.__matches_csv_path,
                                       self.__db_folder,
                                       self.__include_path,
                                       self.__img_paths_csv_path,
                                       self.__transform,
                                       self.__augment)
        self.assertEqual(self.__ds[0]['image'].shape, (3, 224, 224))

    def test_when_grayscale(self):
        self.__ds = DbRetrievalDataset(self.__data_folder,
                                       self.__matches_csv_path,
                                       self.__db_folder,
                                       self.__include_path,
                                       self.__img_paths_csv_path,
                                       self.__transform,
                                       self.__augment,
                                       grayscale=True)

        self.assertEqual(self.__ds[0]['image'].shape, (1, 224, 224))

    def test_input_dtype_when_specified(self):
        self.__input_dtype = 'float32'
        self.__ds = DbRetrievalDataset(self.__data_folder,
                                       self.__matches_csv_path,
                                       self.__db_folder,
                                       self.__include_path,
                                       self.__img_paths_csv_path,
                                       self.__transform,
                                       self.__augment,
                                       input_dtype=self.__input_dtype)
        self.assertEqual(self.__ds[0]['image'].dtype, torch.__dict__[self.__input_dtype])

    def test_output_format(self):
        self.__ds = DbRetrievalDataset(self.__data_folder,
                                       self.__matches_csv_path,
                                       self.__db_folder,
                                       self.__include_path,
                                       self.__img_paths_csv_path,
                                       self.__transform,
                                       self.__augment)
        self.assertListEqual([*self.__ds[0]], ['image', 'index', 'target'])

    def test_shape_targets(self):
        self.__ds = DbRetrievalDataset(self.__data_folder,
                                       self.__matches_csv_path,
                                       self.__db_folder,
                                       self.__include_path,
                                       self.__img_paths_csv_path,
                                       self.__transform,
                                       self.__augment)
        self.assertTupleEqual(self.__ds.targets.shape, (len(self.__ds), self.__ds.n_queries + 1))

    def test_when_augment_none_and_grayscale_true(self):
        self.__ds = DbRetrievalDataset(self.__data_folder,
                                       self.__matches_csv_path,
                                       self.__db_folder,
                                       self.__include_path,
                                       self.__img_paths_csv_path,
                                       self.__transform)
        self.assertTupleEqual(self.__ds[0]['image'].shape, (3, 224, 224))


if __name__ == "__main__":
    unittest.main()
