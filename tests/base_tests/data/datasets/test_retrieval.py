import unittest
from pathlib import Path

import torch

from tests.base_tests.data.datasets.test_image_classification import TestImageDataset
from torchok.data.datasets.representation.validation import RetrievalDataset


class TestRetrievalDataset(TestImageDataset, unittest.TestCase):
    data_cls = RetrievalDataset

    def setUp(self) -> None:
        super().setUp()
        root = Path('tests/base_tests/data/datasets/retrieval_data')
        self.dataset_kwargs['data_folder'] = root
        self.dataset_kwargs['matches_csv_path'] = 'toilets_match.csv'
        self.dataset_kwargs['img_list_csv_path'] = 'toilets_paths.csv'
        self.dataset_kwargs['augment'] = self.augment
        self.ds_len = 12

        self.gallery_folder = root
        self.gallery_list_csv_path = 'toilets_gallery.csv'

    def test_len(self):
        super().test_len()

    def test_shape_when_transformed(self):
        super().test_shape_when_transformed()

    def test_shape_when_grayscale(self):
        super().test_shape_when_grayscale()

    def test_input_dtype(self):
        super().test_input_dtype()

    def test_shape_when_use_gallery_true(self):
        self.dataset_kwargs['gallery_folder'] = self.gallery_folder
        self.dataset_kwargs['gallery_list_csv_path'] = self.gallery_list_csv_path
        ds = self.create_dataset()
        self.assertEqual(ds[18]['image'].shape, (3, 224, 224))

    def test_len_when_use_gallery_true(self):
        self.dataset_kwargs['gallery_folder'] = self.gallery_folder
        self.dataset_kwargs['gallery_list_csv_path'] = self.gallery_list_csv_path
        self.ds_len = 20
        super().test_len()

    def test_output_format(self):
        ds = self.create_dataset()
        self.assertListEqual(list(ds[0].keys()), ['image', 'index', 'query_idxs', 'scores', 'group_labels'])

    def test_group_label_tensor(self):
        ds = self.create_dataset()
        true_target = torch.tensor([0, 0, 1, 1, 0, 1, 2, 0, 2, 2, 1, 3])
        self.assertTrue(torch.equal(ds.group_labels, true_target))

    def test_group_label_tensor_when_use_gallery_true(self):
        self.dataset_kwargs['gallery_folder'] = self.gallery_folder
        self.dataset_kwargs['gallery_list_csv_path'] = self.gallery_list_csv_path
        ds = self.create_dataset()
        true_target = torch.tensor([0, 0, 1, 1, 0, 1, 2, 0, 2, 2, 1, 3, -1, -1, -1, -1, -1, -1, -1, -1])
        self.assertTrue(torch.equal(ds.group_labels, true_target))

    def test_target_tensor_when_gallery_false(self):
        ds = self.create_dataset()

        true_target = torch.tensor([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [4., 0., 0.],
                                    [3., 0., 0.],
                                    [2., 0., 0.],
                                    [0., 4., 0.],
                                    [0., 3., 0.],
                                    [0., 2., 0.],
                                    [0., 0., 4.],
                                    [0., 0., 3.],
                                    [0., 0., 2.]])

        self.assertTrue(torch.equal(ds.scores, true_target))

    def test_target_tensor_when_gallery_true(self):
        self.dataset_kwargs['gallery_folder'] = self.gallery_folder
        self.dataset_kwargs['gallery_list_csv_path'] = self.gallery_list_csv_path
        ds = self.create_dataset()

        true_target = torch.tensor([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [4., 0., 0.],
                                    [3., 0., 0.],
                                    [2., 0., 0.],
                                    [0., 4., 0.],
                                    [0., 3., 0.],
                                    [0., 2., 0.],
                                    [0., 0., 4.],
                                    [0., 0., 3.],
                                    [0., 0., 2.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.]])

        self.assertTrue(torch.equal(ds.scores, true_target))
