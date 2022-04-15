import unittest
from pathlib import Path

from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip

from src.data.datasets.representation.validation import RetrievalDataset


class TestRetrievalDataset(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.__root_dir = Path(__file__).parent
        self.__gallery_folder = self.__root_dir / 'data/retrieval/orig/'
        self.__data_folder = self.__root_dir / 'data/retrieval/orig/'
        self.__gallery_path = self.__root_dir / 'data/retrieval/toilets_gallery.csv'
        self.__img_paths_csv_path = self.__root_dir / 'data/retrieval/toilets_paths.csv'
        self.__matches_csv_path = self.__root_dir / 'data/retrieval/toilets_match.csv'
        self.__transform = Compose([Resize(224, 224), ToTensorV2()], p=1.0)
        self.__augment = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])

    def test_len(self):
        self.__ds = RetrievalDataset(self.__data_folder,
                                     self.__matches_csv_path,
                                     self.__img_paths_csv_path,
                                     self.__transform,
                                     self.__augment)
        self.assertEqual(len(self.__ds), 12)

    def test_shape_when_transformed(self):
        self.__ds = RetrievalDataset(self.__data_folder,
                                     self.__matches_csv_path,
                                     self.__img_paths_csv_path,
                                     self.__transform,
                                     self.__augment)
        self.assertEqual(self.__ds[0]['image'].shape, (3, 224, 224))

    def test_when_grayscale(self):
        self.__ds = RetrievalDataset(self.__data_folder,
                                     self.__matches_csv_path,
                                     self.__img_paths_csv_path,
                                     self.__transform,
                                     self.__augment,
                                     grayscale=True)
        self.assertEqual(self.__ds[0]['image'].shape, (1, 224, 224))

    def test_shape_when_gallery(self):
        self.__ds = RetrievalDataset(self.__data_folder,
                                     self.__matches_csv_path,
                                     self.__img_paths_csv_path,
                                     self.__transform,
                                     self.__augment,
                                     use_gallery=True,
                                     gallery_folder=self.__gallery_folder,
                                     gallery_list_csv_path=self.__gallery_path)
        self.assertEqual(self.__ds[0]['image'].shape, (3, 224, 224))

    def test_len_when_gallery(self):
        self.__ds = RetrievalDataset(self.__data_folder,
                                     self.__matches_csv_path,
                                     self.__img_paths_csv_path,
                                     self.__transform,
                                     self.__augment,
                                     use_gallery=True,
                                     gallery_folder=self.__gallery_folder,
                                     gallery_list_csv_path=self.__gallery_path)
        self.assertEqual(len(self.__ds), 20)

    def test_output_format(self):
        self.__ds = RetrievalDataset(self.__data_folder,
                                     self.__matches_csv_path,
                                     self.__img_paths_csv_path,
                                     self.__transform,
                                     self.__augment)
        self.assertListEqual([*self.__ds[0]], ['image', 'index', 'is_query', 'scores'])


if __name__ == "__main__":
    unittest.main()
