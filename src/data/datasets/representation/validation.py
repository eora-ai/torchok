from pathlib import Path
from typing import Tuple, Union, Optional

import torch
import pandas as pd
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from src.data.datasets.base import ImageDataset


class RetrievalDataset(ImageDataset):
    """Dataset for query->relevant benchmarking
    Dataset for image retrieval where gallery consists of query, relevant and gallery items(optional).
    The searches are made by queries while looking for relevant items in the whole set of items.
    Where gallery items are treated non-relevant.

    Example match.csv:

    Query ids should be unique, otherwise the rows having the same query id will be treated as different matches.
    Relevant ids can be repeated in different queries.
    Scores reflect the order of similarity of the image to the query,
    a higher score corresponds to a greater similarity.

    .. csv-table:: Match csv example
    :header: query, relevant, scores
    1194917,601566 554492 224125 2001716519,4 3 2 2
    1257924,456490,4

    Example img_list.csv:

    img_list.csv maps the id's of query and relevant elements to image paths

    .. csv-table:: Image csv example
    :header: id, image_path
    1194917,data/img_1.jpg
    601566,data/img_2.jpg
    554492,data/img_3.jpg
    224125,data/img_4.jpg
    2001716519,data/img_5.jpg
    1257924,data/img_6.jpg
    456490,data/img_7.jpg

    .. csv-table:: Gallery Image csv example
    :header: id, image_paths
    8,data/db/img_1.jpg
    10,data/db/img_2.jpg
    12,data/db/img_3.jpg
    """

    def __init__(self,
                 data_folder: str,
                 matches_csv_path: str,
                 img_list_csv_path: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 use_gallery: Optional[bool] = False,
                 gallery_folder: Optional[str] = None,
                 gallery_list_csv_path: Optional[str] = None,
                 input_dtype: str = 'float32',
                 input_column: str = 'image_path',
                 grayscale: bool = False):
        """
        Args:
            data_folder: Directory with all the images.
            matches_csv_path: path to csv file where queries with their relevance scores are specified
            img_list_csv_path: path to mapping image identifiers to image paths. Format: id | path.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            use_gallery: If True, will use gallery data(non-relevant items)
            gallery_folder: path to a folder with all gallery images (traversed recursively)
            gallery_list_csv_path: path to mapping image identifiers to image paths. Format: id | path.
            input_dtype: Data type of of the torch tensors related to the image.
            input_column: Name of the column that contains paths to images.
            grayscale: If True, image will be read as grayscale otherwise as RGB.

        Raises:
            ValueError: if use_gallery True, but gallery_folder or gallery_list_csv_path is None
        """
        super().__init__(data_folder, transform, augment, input_dtype, input_column, grayscale)

        self.__matches = pd.read_csv(self._data_folder / matches_csv_path,
                                     usecols=['query', 'relevant', 'scores'],
                                     dtype={'query': int, 'relevant': str, 'scores': str})

        self.__img_paths = pd.read_csv(self._data_folder / img_list_csv_path,
                                       usecols=['id', self._input_column],
                                       dtype={'id': int, self._input_column: str})

        self.__parse_match_csv()

        self._imgid2paths = dict(zip(self.__img_paths['id'], self.__img_paths[self._input_column]))
        # filtering (save only img_id that in match.csv)
        self._imgid2paths = {img_id: self._imgid2paths[img_id] for img_id in self.__index2imgid.values()}
        self._data_len = self.__n_queries + self.__n_relevant

        self.__use_gallery = use_gallery

        if self.__use_gallery:
            self.__gallery_folder = gallery_folder
            self.__gallery_list_csv_path = gallery_list_csv_path

            if self.__gallery_folder is None:
                raise ValueError('Argument `gallery_folder` is None, please send path to gallery_folder')
            if self.__gallery_list_csv_path is None:
                raise ValueError('Argument `gallery_list_csv_path` is None, please send path to gallery_list_csv_path')

            self.__gallery_paths = pd.read_csv(Path(self.__gallery_folder) / self.__gallery_list_csv_path,
                                               usecols=['id', self._input_column],
                                               dtype={'id': int, self._input_column: str})
            self.__gallery_imgid2paths = dict(zip(self.__gallery_paths['id'], self.__gallery_paths[self._input_column]))

            self.__n_gallery = 0
            self._gallery_index2imgid = {}
            for img_id in self.__gallery_imgid2paths:
                self._gallery_index2imgid[self.__n_gallery] = img_id
                self.__n_gallery += 1

            self._data_len += self.__n_gallery

        self.__imgid2index = {}
        for index, img_id in self.__index2imgid.items():
            if img_id not in self.__imgid2index:
                self.__imgid2index[img_id] = index

        self.__scores, self.__is_query = self.__get_targets()

    def __getitem__(self, index: int) -> dict:

        if index < self.__n_queries + self.__n_relevant:
            img_id = self.__index2imgid[index]
            image_path = self._data_folder / self._imgid2paths[img_id]
        else:
            img_id = self._gallery_index2imgid[index]
            image_path = self.__gallery_folder / self.__gallery_imgid2paths[img_id]

        image = self._read_image(image_path)
        sample = {'image': image}
        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self._input_dtype])
        sample['index'] = index
        sample['is_query'] = self.__is_query[index]
        sample['scores'] = self.__scores[index]

        return sample

    def __parse_match_csv(self):
        self.__query_arr = self.__matches.loc[:, 'query'].tolist()
        self.__index2imgid = dict(enumerate(self.__query_arr))
        self.__n_queries = len(self.__index2imgid)

        self.__relevant_arr, self.__relevance_scores = [], []
        self.__n_relevant = 0

        for index in range(len(self.__matches)):
            self.__relevant_arr.append([])
            self.__relevance_scores.append([])
            rel_img_idxs = map(int, self.__matches.iloc[index]['relevant'].split())
            rel_img_scores = map(float, self.__matches.iloc[index]['scores'].split())
            for img_id, img_score in zip(rel_img_idxs, rel_img_scores):
                # save unique image_id
                if img_id not in self.__index2imgid.values():
                    self.__index2imgid[self.__n_queries + self.__n_relevant] = img_id
                    self.__n_relevant += 1
                self.__relevant_arr[-1].append(img_id)
                self.__relevance_scores[-1].append(img_score)

    def __get_targets(self) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """
        Maps item scores to queues.

        Returns:
            Two target tensor: scores and is_query.
            Scores is tensor with shape(len(self), n_queries).
            Is_query is tensor with shape (len(self)).
        Raises:
            ValueError: If relevant objects list doesn't match with relevance scores list in size.
        """

        scores = torch.zeros((len(self), self.__n_queries), dtype=torch.float32)
        is_query = torch.zeros(len(self), dtype=torch.bool)

        for index in range(self.__n_queries):
            relevant_img_idxs = self.__relevant_arr[index]
            relevance_scores = self.__relevance_scores[index]
            if len(relevant_img_idxs) != len(relevance_scores):
                raise ValueError(f"Relevant objects list must match relevance scores list in size."
                                 f"Got number of relevant object indices: {len(relevant_img_idxs)}, "
                                 f"number of relevance scores: {len(relevance_scores)}")

            relevant_indices = [self.__imgid2index[img_id] for img_id in relevant_img_idxs]
            for rel_index, score in zip(relevant_indices, relevance_scores):
                scores[rel_index][index] = score
            is_query[index] = True

        return scores, is_query

    def __len__(self) -> int:
        return self._data_len

    @property
    def matches(self) -> pd.DataFrame:
        return self.__matches

    @property
    def img_paths(self) -> pd.DataFrame:
        return self.__img_paths

    @property
    def use_gallery(self) -> bool:
        return self.__use_gallery

    @property
    def gallery_list_csv_path(self) -> Optional[str]:
        return self.__gallery_list_csv_path

    @property
    def gallery_folder(self) -> Optional[str]:
        return self.__gallery_folder

    @property
    def gallery_paths(self) -> Optional[pd.DataFrame]:
        return self.__gallery_paths

    @property
    def n_queries(self) -> int:
        return self.__n_queries

    @property
    def n_relevant(self) -> int:
        return self.__n_relevant

    @property
    def n_gallery(self) -> Optional[int]:
        return self.__n_gallery
