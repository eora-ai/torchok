from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import pandas as pd

from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from torchok.constructor import DATASETS
from torchok.data.datasets.base import ImageDataset


@DATASETS.register_class
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
        :header: id, image_path, label

        1194917,data/img_1.jpg,0
        601566,data/img_2.jpg,0
        554492,data/img_3.jpg,0
        224125,data/img_4.jpg,1
        2001716519,data/img_5.jpg,1
        1257924,data/img_6.jpg,1
        456490,data/img_7.jpg,2

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
                 gallery_folder: Optional[str] = '',
                 gallery_list_csv_path: Optional[str] = None,
                 use_query_without_relevants: bool = False,
                 input_dtype: str = 'float32',
                 channel_order: str = 'rgb',
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
            use_query_without_relevants: If True, use query without relevants.
            input_dtype: Data type of the torch tensors related to the image.
            channel_order: Order of channel, candidates are `bgr` and `rgb`.
            grayscale: If True, image will be read as grayscale otherwise as RGB.

        Raises:
            ValueError: if gallery_folder `True`, but `gallery_list_csv_path` is `None`.
        """
        super().__init__(
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            channel_order=channel_order,
            grayscale=grayscale
        )
        self.data_folder = Path(data_folder)
        self.use_query_without_relevants = use_query_without_relevants
        matches_dtype = {'query': int, 'relevant': str, 'scores': str}
        img_paths_dtype = {'img_id': int, 'image_path': str, 'label': int}

        self.matches = pd.read_csv(self.data_folder / matches_csv_path, dtype=matches_dtype)
        self.img_paths = pd.read_csv(self.data_folder / img_list_csv_path,
                                     dtype=img_paths_dtype)

        self.use_scores = 'scores' in self.matches.columns
        self.use_group_labels = 'label' in self.img_paths.columns

        self.n_not_query, self.n_queries, self.index2imgid, self.imgid2index,\
            self.index2label, self.relevant_arr, self.relevance_scores = self._parse_match_csv()

        self.imgid2path = dict(zip(self.img_paths['id'],
                                   self.img_paths['image_path']))

        if len(self.imgid2path) != len(self.img_paths):
            raise ValueError('Image csv have the same id for different image paths.')

        self.data_len = self.n_queries + self.n_not_query

        if gallery_list_csv_path is not None:
            self.gallery_folder = Path(gallery_folder)
            self.gallery_list_csv_path = gallery_list_csv_path
            if self.gallery_list_csv_path is None:
                raise ValueError('Argument `gallery_list_csv_path` is None, please send path to gallery_list_csv_path.')

            gallery_paths_dtype = {'id': int, 'image_path': str}
            self.gallery_paths = pd.read_csv(self.gallery_folder / self.gallery_list_csv_path,
                                             dtype=gallery_paths_dtype)
            self.gallery_imgid2path = dict(zip(self.gallery_paths['id'],
                                               self.gallery_paths['image_path']))

            if len(self.gallery_imgid2path) != len(self.gallery_paths):
                raise ValueError('Gallery csv have the same id for different image paths.')

            self.n_gallery = 0
            self.gallery_index2imgid = {}
            for img_id in self.gallery_imgid2path:
                self.gallery_index2imgid[self.data_len + self.n_gallery] = img_id
                self.n_gallery += 1

            self.data_len += self.n_gallery

        self.scores, self.query_idxs, self.group_labels = self._get_targets()

    def get_raw(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            dict with fields:
                `image` - np.array, representing image after augmentations, dtype=input_dtype.
                `index` - Index from DataFrame.
                `query_idxs` - Int tensor, if item is query: return index of this query in target matrix, else -1.
                `scores` - Float tensor shape (1, len(n_query)), relevant scores of current item.
                `group_labels` - Int tensor with image classification label.
        """
        if idx < self.n_queries + self.n_not_query:
            img_id = self.index2imgid[idx]
            image_path = self.data_folder / self.imgid2path[img_id]
        else:
            img_id = self.gallery_index2imgid[idx]
            image_path = self.gallery_folder / self.gallery_imgid2path[img_id]

        image = self._read_image(image_path)
        sample = {'image': image, 'index': idx, 'query_idxs': self.query_idxs[idx],
                  'scores': self.scores[idx], 'group_labels': self.group_labels[idx]}
        return self._apply_transform(self.augment, sample)

    def __getitem__(self, index: int) -> dict:
        """Get item sample.

        Returns:
            dict with fields:
                `image` - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
                `index` - Index from DataFrame.
                `query_idxs` - Int tensor, if item is query: return index of this query in target matrix, else -1.
                `scores` - Float tensor shape (1, len(n_query)), relevant scores of current item.
                `group_labels` - Int tensor with image classification label.
        """
        sample = self.get_raw(index)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        return sample

    def _parse_match_csv(self) -> Tuple[int, int, dict, dict, dict, list, list]:
        """Parse input csvs into needed data.

        Returns:
            n_not_query: Number of images in self.img_paths which is not query.
            n_queries: Number of images in self.img_paths which is query.
            index2imgid: Index to image id dict, where image id is id from self.img_paths.
            imgid2index: Image id to index, where image id is id from self.img_paths (same as index2imgid).
            index2label: Index to label dict (index same for the index2imgid and imgid2index).
            relevant_arr: Array of relevant image id for each query.
            relevance_scores: Array of relevant scores for each query.
        """
        query_arr = self.matches.loc[:, 'query'].tolist()
        index2imgid = dict(enumerate(query_arr))
        imgid2index = dict(zip(query_arr, range(len(query_arr))))
        n_queries = len(index2imgid)

        relevant_arr, relevance_scores = [], []
        n_not_query = 0

        for index in range(len(self.matches)):
            row_relevants, row_scores = [], []
            # if no relevants add empty list to relevants
            if pd.isna(self.matches.iloc[index]['relevant']):
                if self.use_query_without_relevants:
                    relevant_arr.append(list())
                    relevance_scores.append(list())
                    continue
                else:
                    raise ValueError('Match csv has query without relevant elements. Check your csv or set '
                                     'parameter use_query_without_relevants=True to set relevants as empty '
                                     'for these queries.')

            rel_img_idxs = list(map(int, self.matches.iloc[index]['relevant'].split()))

            if self.use_scores:
                rel_img_scores = list(map(float, self.matches.iloc[index]['scores'].split()))
            else:
                rel_img_scores = [1] * len(rel_img_idxs)

            if len(rel_img_idxs) != len(rel_img_scores):
                raise ValueError(f'Relevant objects list must match relevance scores list in size.'
                                 f'Got number of relevant object indices: {len(rel_img_idxs)}, '
                                 f'number of relevance scores: {len(rel_img_scores)}')

            for img_id, img_score in zip(rel_img_idxs, rel_img_scores):
                # save unique image_id
                if img_id not in imgid2index:
                    index2imgid[n_queries + n_not_query] = img_id
                    imgid2index[img_id] = n_queries + n_not_query
                    n_not_query += 1
                row_relevants.append(img_id)
                row_scores.append(img_score)

            relevant_arr.append(row_relevants)
            relevance_scores.append(row_scores)

        for img_id in self.img_paths.id:
            if img_id not in imgid2index:
                index2imgid[n_queries + n_not_query] = img_id
                imgid2index[img_id] = n_queries + n_not_query
                n_not_query += 1

        index2label = dict()
        for index, img_id in index2imgid.items():
            label = self.img_paths.loc[self.img_paths.id == img_id].iloc[0]['label'] if self.use_group_labels else 0
            index2label[index] = label

        return n_not_query, n_queries, index2imgid, imgid2index, index2label, relevant_arr, relevance_scores

    def _get_targets(self) -> Tuple[torch.FloatTensor, torch.IntTensor, torch.IntTensor]:
        """Mapping item scores to queues.

        Returns:
            Three target tensor: scores, query_idxs and group_labels.
            Scores is tensor with shape: (len(self), n_queries).
            Query_idxs is tensor with shape: (len(self)).
            group_labels is tensor with shape: (len(self)).

        Raises:
            ValueError: If relevant objects list doesn't match with relevance scores list in size.
        """
        scores = torch.zeros((len(self), self.n_queries), dtype=torch.float32)
        query_idxs = torch.full((len(self),), -1, dtype=torch.int32)
        group_labels = torch.full((len(self),), -1, dtype=torch.long)

        for index in range(self.n_queries):
            relevant_img_idxs = self.relevant_arr[index]
            relevance_scores = self.relevance_scores[index]
            relevant_indices = [self.imgid2index[img_id] for img_id in relevant_img_idxs]
            for rel_index, score in zip(relevant_indices, relevance_scores):
                scores[rel_index][index] = score
            query_idxs[index] = index

        for index, label in self.index2label.items():
            group_labels[index] = label

        return scores, query_idxs, group_labels

    def __len__(self) -> int:
        """Length of Retrieval dataset."""
        return self.data_len
