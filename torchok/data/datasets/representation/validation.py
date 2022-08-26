from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import pandas as pd

from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.data.datasets.base import ImageDataset


class RetrievalDataset(ImageDataset):
    """Dataset for image retrieval validation.

    The searches are made by queries while looking for relevant items in the whole set of items.
    Where gallery items are treated non-relevant.

    Example matches csv:
    Query ids should be unique int values,
    otherwise the rows having the same query id will be treated as different matches.

    Relevant ids can be repeated in different queries.

    Scores reflect the order of similarity of the image to the query,
    a higher score corresponds to a greater similarity(must be float value > 0.).

    .. csv-table:: Match csv example
        :header: query, relevant, scores

        1194917,601566 554492 224125 2001716519,4 3 2 2
        1257924,456490,4

    Example img_list csv:
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
                 gallery_folder: Optional[str] = None,
                 gallery_list_csv_path: Optional[str] = None,
                 input_dtype: str = 'float32',
                 img_list_map_column: dict = None,
                 matches_map_column: dict = None,
                 gallery_map_column: dict = None,
                 grayscale: bool = False):
        """Init RetrievalDataset class.

        Args:
            data_folder: Directory with all the images.
            matches_csv_path: path to csv file where queries with their relevance scores are specified
            img_list_csv_path: path to mapping image identifiers to image paths. Format: id | path.
                ID from matches csv are linked to id from img_list csv
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            gallery_folder: Path to a folder with all gallery images (traversed recursively).
                            When the gallery not specified all the remaining queries and relevant
                            will be considered as negative samples to a given query-relevant set.
            gallery_list_csv_path: Path to mapping image identifiers to image paths. Format: id | path.
            input_dtype: Data type of the torch tensors related to the image.
            img_list_map_column: Image mapping column names. Key - TorchOk column name, Value - csv column name.
                default value: {'image_path': 'image_path', 'img_id': 'id'}
            matches_map_column: Matches mapping column names. Key - TorchOk column name, Value - csv column name.
                default value: {'query': 'query', 'relevant': 'relevant', 'scores': 'scores'}
            gallery_map_column: Gallery mapping column names. Key - TorchOk column name, Value - csv column name.
                default value: {'gallery_path': 'image_path', 'gallery_id': 'id'}
            grayscale: If True, image will be read as grayscale otherwise as RGB.

        Raises:
            ValueError: if gallery_folder True, but gallery_list_csv_path is None
        """
        super().__init__(transform, augment, input_dtype, grayscale)
        self.data_folder = Path(data_folder)
        self.matches_map_column = matches_map_column or {'query': 'query', 'relevant': 'relevant', 'scores': 'scores'}
        self.img_list_map_column = img_list_map_column or {'image_path': 'image_path', 'img_id': 'id'}
        self.gallery_map_column = gallery_map_column or {'gallery_path': 'image_path', 'gallery_id': 'id'}

        self.matches = pd.read_csv(self.data_folder / matches_csv_path,
                                   usecols=[self.matches_map_column['query'],
                                            self.matches_map_column['relevant'],
                                            self.matches_map_column['scores']],
                                   dtype={self.matches_map_column['query']: int,
                                          self.matches_map_column['relevant']: str,
                                          self.matches_map_column['scores']: str})

        self.img_paths = pd.read_csv(self.data_folder / img_list_csv_path,
                                     usecols=[self.img_list_map_column['img_id'],
                                              self.img_list_map_column['image_path']],
                                     dtype={self.img_list_map_column['img_id']: int,
                                            self.img_list_map_column['image_path']: str})

        self.n_relevant, self.n_queries, self.index2imgid,\
            self.relevant_arr, self.relevance_scores = self._parse_match_csv()

        self.imgid2path = dict(zip(self.img_paths[self.img_list_map_column['img_id']],
                                   self.img_paths[self.img_list_map_column['image_path']]))

        if len(self.imgid2path) != len(self.img_paths):
            raise ValueError('Image csv have the same id for different image paths')

        # filtering (save only img_id that in match.csv)
        self.imgid2path = {img_id: self.imgid2path[img_id] for img_id in self.index2imgid.values()}
        self.data_len = self.n_queries + self.n_relevant

        if gallery_folder is not None:
            self.gallery_folder = Path(gallery_folder)
            self.gallery_list_csv_path = gallery_list_csv_path
            if self.gallery_list_csv_path is None:
                raise ValueError('Argument `gallery_list_csv_path` is None, please send path to gallery_list_csv_path')

            self.gallery_paths = pd.read_csv(self.gallery_folder / self.gallery_list_csv_path,
                                             usecols=[self.gallery_map_column['gallery_id'],
                                                      self.gallery_map_column['gallery_path']],
                                             dtype={self.gallery_map_column['gallery_id']: int,
                                                    self.gallery_map_column['gallery_path']: str})
            self.gallery_imgid2path = dict(zip(self.gallery_paths[self.gallery_map_column['gallery_id']],
                                               self.gallery_paths[self.gallery_map_column['gallery_path']]))

            if len(self.gallery_imgid2path) != len(self.gallery_paths):
                raise ValueError('Gallery csv have the same id for different image paths')

            self.n_gallery = 0
            self.gallery_index2imgid = {}
            for img_id in self.gallery_imgid2path:
                self.gallery_index2imgid[self.data_len + self.n_gallery] = img_id
                self.n_gallery += 1

            self.data_len += self.n_gallery

        self.imgid2index = {}
        for index, img_id in self.index2imgid.items():
            if img_id not in self.imgid2index:
                self.imgid2index[img_id] = index

        self.scores, self.is_query = self._get_targets()

    def get_raw(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - np.array, representing image after augmentations, dtype=input_dtype.
            sample['index'] - Index.
            sample['is_query'] - Int tensor, if item is query: return index of this query in target matrix, else -1.
            sample['scores'] - Float tensor shape (1, len(n_query)), relevant scores of current item.
        """
        if idx < self.n_queries + self.n_relevant:
            img_id = self.index2imgid[idx]
            image_path = self.data_folder / self.imgid2path[img_id]
        else:
            img_id = self.gallery_index2imgid[idx]
            image_path = self.gallery_folder / self.gallery_imgid2path[img_id]

        image = self._read_image(image_path)
        sample = {'image': image, 'index': idx, 'is_query': self.is_query[idx], 'scores': self.scores[idx]}
        return self._apply_transform(self.augment, sample)

    def __getitem__(self, index: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['index'] - Index.
            sample['is_query'] - Int tensor, if item is query: return index of this query in target matrix, else -1.
            sample['scores'] - Float tensor shape (1, len(n_query)), relevant scores of current item.
        """
        sample = self.get_raw(index)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        return sample

    def _parse_match_csv(self) -> Tuple[int, int, dict, list, list]:
        query_arr = self.matches.loc[:, 'query'].tolist()
        index2imgid = dict(enumerate(query_arr))
        n_queries = len(index2imgid)

        relevant_arr, relevance_scores = [], []
        n_relevant = 0

        for index in range(len(self.matches)):
            row_relevants, row_scores = [], []
            rel_img_idxs = list(map(int, self.matches.iloc[index]['relevant'].split()))
            rel_img_scores = list(map(float, self.matches.iloc[index]['scores'].split()))

            if len(rel_img_idxs) != len(rel_img_scores):
                raise ValueError(f'Relevant objects list must match relevance scores list in size.'
                                 f'Got number of relevant object indices: {len(rel_img_idxs)}, '
                                 f'number of relevance scores: {len(rel_img_scores)}')

            for img_id, img_score in zip(rel_img_idxs, rel_img_scores):
                # save unique image_id
                if img_id not in index2imgid.values():
                    index2imgid[n_queries + n_relevant] = img_id
                    n_relevant += 1
                row_relevants.append(img_id)
                row_scores.append(img_score)

            relevant_arr.append(row_relevants)
            relevance_scores.append(row_scores)
        return n_relevant, n_queries, index2imgid, relevant_arr, relevance_scores

    def _get_targets(self) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """Maping item scores to queues.

        Returns:
            Two target tensor: scores and is_query.
            Scores is tensor with shape: (len(self), n_queries).
            Is_query is tensor with shape: (len(self)).

        Raises:
            ValueError: If relevant objects list doesn't match with relevance scores list in size.
        """
        scores = torch.zeros((len(self), self.n_queries), dtype=torch.float32)
        is_query = torch.full((len(self),), -1, dtype=torch.int32)

        for index in range(self.n_queries):
            relevant_img_idxs = self.relevant_arr[index]
            relevance_scores = self.relevance_scores[index]
            relevant_indices = [self.imgid2index[img_id] for img_id in relevant_img_idxs]
            for rel_index, score in zip(relevant_indices, relevance_scores):
                scores[rel_index][index] = score
            is_query[index] = index

        return scores, is_query

    def __len__(self) -> int:
        """Length of Retrieval dataset."""
        return self.data_len
