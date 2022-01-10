from pathlib import Path
from typing import Union, List, Dict, Any, Optional

from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate

from src.registry import DATASETS
from .abc_dataset import ABCDataset


class QueryToRelevantDataset(ABCDataset):
    def __init__(self, data_folder: str, matches_csv_path: str, img_paths_csv_path: str,
                 retrieval_level: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32', grayscale: bool = False, seq_len: int = None):
        """
        Base dataset for query->relevant benchmarking
        :param matches_csv_path: path to csv file where queries with their relevance scores are specified
        :param data_folder: path to data folder where images and img_paths.csv are present
        :param img_paths_csv_path: path to mapping image identifiers to image paths. Format: id | path.
        Default: img_paths.csv
        """
        super().__init__(transform, augment)
        self._data_folder = Path(data_folder)
        self._matches = pd.read_csv(self._data_folder / matches_csv_path, dtype={"query": str, "relevant": str})
        self.input_dtype = input_dtype
        self.grayscale = grayscale
        if retrieval_level == 'image':
            self._img_paths = pd.read_csv(self._data_folder / img_paths_csv_path, usecols=["id", "path"],
                                          dtype={"id": str, "path": str}, header=0)
        elif retrieval_level == 'object':
            if seq_len is None:
                raise ValueError(f"if retrieval_level is 'object', needed to provide 'seq_len'. Got: {seq_len}")

            self._img_paths = pd.read_csv(self._data_folder / img_paths_csv_path, usecols=["id", "is_mainview", "path"],
                                          dtype={"id": str, "path": str, "is_mainview": int}, header=0)
        else:
            raise ValueError(f"retrieval_level must be either 'object' or 'image'. Got: {retrieval_level}")
        self._img_paths['path'] = self._img_paths['path'].map(lambda p: self._data_folder / p)

        self.seq_len = seq_len
        self.retrieval_level = retrieval_level
        self.update_transform_targets({'input': 'image'})

        self._query_arr = self._matches.loc[:, "query"].tolist()
        self._index2objid = dict(enumerate(self._query_arr))
        self._n_queries = len(self._index2objid)

        self._relevant_arr, self._relevance_scores = [], []
        n_relevant = 0
        for index in range(len(self._matches)):
            self._relevant_arr.append([])
            self._relevance_scores.append([])
            rel_obj_idxs = self._matches.iloc[index]["relevant"].split()
            rel_obj_scores = map(float, self._matches.iloc[index]["scores"].split())
            for obj_id, obj_score in zip(rel_obj_idxs, rel_obj_scores):
                self._index2objid[self._n_queries + n_relevant] = obj_id
                n_relevant += 1
                self._relevant_arr[-1].append(obj_id)
                self._relevance_scores[-1].append(obj_score)
        self._n_relevant = n_relevant

        objid2paths = self._img_paths.groupby("id")["path"].apply(list).to_dict()
        self._objid2paths = {obj_id: objid2paths[obj_id] for obj_id in self._index2objid.values()}
        if self.retrieval_level == 'object':
            mainviews = self._img_paths[self._img_paths["is_mainview"] == 1]
            objid2mainviewpath = mainviews.groupby("id")["path"].first().to_dict()
            self._objid2mainviewpath = {obj_id: objid2mainviewpath[obj_id] for obj_id in self._index2objid.values()}
        self._data_len = self._n_queries + self._n_relevant

    def read_image(self, image_path) -> np.ndarray:
        with Image.open(image_path) as image:
            image = np.array(image.convert('RGB'))

        # FIXME: check the other cases
        # for some reason some TIF images are read as binary, fix them
        if image.dtype == np.bool8:
            image = (image * 255).astype(np.uint8)

        if image.ndim > 2 and image.shape[2] == 4:
            img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.ndim == 2:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img = image

        return img

    def get_raw(self, obj_id: str):
        """
        Get initial image as is from database
        :param obj_id: object id
        :return: image of shape (H, W, 3) in RGB order
        """
        item_paths = self._objid2paths[obj_id]
        if self.retrieval_level == 'image':
            # take first path of the given object
            item_path = item_paths[0]
            image = self.read_image(item_path)
            sample = {"input": image}
            sample = self.apply_transform(self.augment, sample)

            return sample
        else:
            mainview_path = self._objid2mainviewpath[obj_id]
            item_paths = [path for path in item_paths if path != mainview_path]
            images = [self.read_image(mainview_path)]
            images.extend([self.read_image(path) for path in item_paths[:min(self.seq_len - 1, len(item_paths))]])
            samples = [self.apply_transform(self.augment, {"input": img}) for img in images]

            return samples

    def __getitem__(self, index: int) -> Dict[str, Union[int, Any]]:
        """
        Get the dataset item by index
        :param index: index of item (row of dataset)
        :return: dictionary
                "image": image of shape (H, W, 3) in RGB order,
                "object_id": int indicating mapped object_id
        """
        sample = self.get_raw(self._index2objid[index])
        if isinstance(sample, dict):
            sample = self.apply_transform(self.transform, sample)
            sample['input'] = sample['input'].type(torch.__dict__[self.input_dtype])
        else:
            input_tensors = [self.apply_transform(self.transform, s)['input'] for s in sample]
            input_tensors = torch.stack(input_tensors)
            sample = {'input': input_tensors}

        sample['object_id'] = index

        return sample

    def __len__(self) -> int:
        return self._data_len

    def get_queries_pivot(self):
        return self._n_queries

    def get_relevant_pivot(self):
        return self._n_queries + self._n_relevant

    @staticmethod
    def collate_fn(batch: dict, field: str = 'input') -> dict:
        """
        Squeezes batch of image sequences with varying length to batch of images,
        keeping the initial length for further unsqueezing
        :param batch:
        :param field: field name for which to squeeze the tensor
        :return:'input' - torch.Tensor of shape (bs, img_size)
                'target' - torch.Tensor, one-hot vector of length num_classes
                'lengths' - torch.Tensor of shape (bs)
        """
        # get sequence lengths
        lengths = []
        input = []
        new_batch = []
        for t in batch:
            inp = t.pop(field)
            input.extend(inp)
            new_batch.append(t)
            lengths.append(inp.shape[0])

        batch = default_collate(new_batch)
        batch[field] = torch.stack(input, 0)
        batch['lengths'] = torch.Tensor(lengths)
        return batch


@DATASETS.register_class
class FullDbRetrievalDataset(QueryToRelevantDataset):
    def __init__(self, data_folder: str, matches_csv_path: str,
                 db_folder: str, include_only_path: str, img_paths_csv_path: str,
                 retrieval_level: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32', seq_len: int = None):
        """
        Dataset for full database image retrieval. It's treated that queries and relevant items
        are separate from the database items
        :param matches_csv_path: path to csv file where queries with their relevance scores are specified
        :param data_folder: path to data folder where images and img_paths.csv are present
        :param db_folder: path to a folder with all database images (traversed recursively)
        :param img_paths_csv_path: path to mapping image identifiers to image paths. Format: id | path.
        Default: img_paths.csv
        :param seq_len: maximum number of images per object to be fitted to the model
        only used if in object retrieval level
        """
        super().__init__(data_folder=data_folder, matches_csv_path=matches_csv_path,
                         img_paths_csv_path=img_paths_csv_path, retrieval_level=retrieval_level,
                         transform=transform, augment=augment, input_dtype=input_dtype, seq_len=seq_len)
        self._db_folder = Path(db_folder)

        if retrieval_level == 'image':
            include_only = pd.read_csv(Path(db_folder) / include_only_path, usecols=["id", "path"],
                                          dtype={"id": str, "path": str}, header=0)
        elif retrieval_level == 'object':
            if seq_len is None:
                raise ValueError(f"if retrieval_level is 'object', needed to provide 'seq_len'. Got: {seq_len}")

            include_only = pd.read_csv(Path(db_folder) / include_only_path, usecols=["id", "is_mainview", "path"],
                                          dtype={"id": str, "path": str, "is_mainview": int}, header=0)
        else:
            raise ValueError(f"retrieval_level must be either 'object' or 'image'. Got: {retrieval_level}")

        print(f"Reading database items from: {include_only_path}")
        include_only['path'] = include_only['path'].map(lambda p: self._db_folder / p)
        db_objid2paths = include_only.groupby('id')['path'].apply(list).to_dict()
        self._objid2paths.update(db_objid2paths)

        if retrieval_level == 'object':
            mainviews = include_only[include_only["is_mainview"] == 1]
            db_objid2mainviewpaths = mainviews.groupby('id')['path'].first().to_dict()
            self._objid2mainviewpath.update(db_objid2mainviewpaths)

        n_db = 0
        for obj_id in db_objid2paths.keys():
            self._index2objid[self.get_relevant_pivot() + n_db] = obj_id
            n_db += 1
        self._n_db = n_db
        self._data_len = self._n_queries + self._n_relevant + self._n_db
        self._objid2index = {}
        for index, obj_id in self._index2objid.items():
            if obj_id not in self._objid2index:
                self._objid2index[obj_id] = index

        self.relevance_map = self.get_relevance_scores()
        self.targets = np.zeros((len(self), 1 + self._n_queries), dtype=np.float32)
        for index in range(self.get_queries_pivot()):
            relevant_idxs = self.relevance_map[index]['relevant_idxs']
            relevant_scores = self.relevance_map[index]['relevant_scores']
            self.targets[relevant_idxs, 1 + index] = relevant_scores
            self.targets[index, 0] = 1.  # query indicator

    def __getitem__(self, index: int) -> Dict[str, Union[int, Any]]:
        sample = super().__getitem__(index)
        sample['target'] = self.targets[index]

        return sample

    def get_relevance_scores(self) -> List[Dict[str, np.ndarray]]:
        """
        Traverses all the query->relevant lists and constructs a map of relevance scores and their positions
        in the joined set of relevant items to each query
        :return: list of relevance cards to each of N queries.
        For each query the following keys are specified:
        - relevance_idxs: indices of relevant embeddings from the joint relevant_embeddings matrix
        - relevance_scores: relevance scores for all of the relevant items
        """
        relevance_map = []

        for index in range(len(self._query_arr)):
            # Read relevant image indices and scores for this query
            relevant_obj_idxs = self._relevant_arr[index]
            relevance_scores = self._relevance_scores[index]
            if len(relevant_obj_idxs) != len(relevance_scores):
                raise ValueError(f"Relevant objects list must match relevance scores list in size."
                                 f"Got number of relevant object indices: {len(relevant_obj_idxs)}, "
                                 f"number of relevance scores: {len(relevance_scores)}")

            relevant_indices = [self._objid2index[obj_id] for obj_id in relevant_obj_idxs]
            relevant_indices = np.array(relevant_indices)

            relevance_map.append({
                'relevant_idxs': relevant_indices,
                'relevant_scores': relevance_scores
            })

        return relevance_map

    @staticmethod
    def collate_fn(batch: dict) -> dict:
        """
        Squeezes batch of image sequences with varying length to batch of images,
        keeping the initial length for further unsqueezing
        :param batch:
        :return:'input' - torch.Tensor of shape (bs, img_size)
                'target' - torch.Tensor, one-hot vector of length num_classes
                'lengths' - torch.Tensor of shape (bs)
        """
        # get sequence lengths
        lengths = []
        inputs = []
        new_batch = []
        for t in batch:
            inp = t.pop('input')
            inputs.extend(inp)
            new_batch.append(t)
            lengths.append(inp.shape[0])

        batch = default_collate(new_batch)
        batch['input'] = torch.stack(inputs, 0)
        batch['lengths'] = torch.Tensor(lengths)
        return batch
