from typing import Union, Optional

import torch
import pandas as pd
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose


from src.data.datasets.base import ImageDataset


class EachToEachRetrievalDataset(ImageDataset):
    """ Dataset for query->relevant benchmarking

     .. csv-table:: Match csv example
     :header: query, relevant, scores
     1194917,601566 554492 224125 2001716519,4 3 2 2
     1257924,456490,4

     .. csv-table:: Image csv example
     :header: id,image_path
     1194917,data/img_1.jpg
     601566,data/img_2.jpg
     554492,data/img_3.jpg
     224125,data/img_4.jpg
     2001716519,data/img_5.jpg
     1257924,data/img_6.jpg
     456490,data/img_7.jpg

    """

    def __init__(self,
                 data_folder: str,
                 matches_csv_path: str,
                 img_paths_csv_path: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 input_column: str = 'image_path',
                 grayscale: bool = False):
        """
        Args:
            data_folder: Directory with all the images.
            matches_csv_path: path to csv file where queries with their relevance scores are specified
            img_paths_csv_path: path to mapping image identifiers to image paths. Format: id | `input_column`.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of of the torch tensors related to the image.
            input_column: Name of the column that contains paths to images.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
        """
        super().__init__(data_folder, transform, augment, input_dtype, input_column, grayscale)

        self.__matches = pd.read_csv(self.data_folder / matches_csv_path,
                                     dtype={"query": str, "relevant": str, 'scores': str})

        self.__img_paths = pd.read_csv(self.data_folder / img_paths_csv_path,
                                       usecols=["id", input_column],
                                       dtype={"id": str, input_column: str},
                                       header=0)

        self.__img_paths[self._input_column] = self.__img_paths[self._input_column]\
                                                   .map(lambda path: self.data_folder / path)

        self.__query_arr = self.__matches.loc[:, "query"].tolist()
        self._index2objid = dict(enumerate(self.__query_arr))
        self._n_queries = len(self._index2objid)

        self._relevant_arr, self._relevance_scores = [], []
        self.__n_relevant = 0

        for index in range(len(self.__matches)):
            self._relevant_arr.append([])
            self._relevance_scores.append([])
            rel_obj_idxs = self.__matches.iloc[index]["relevant"].split()
            rel_obj_scores = map(float, self.__matches.iloc[index]["scores"].split())
            for obj_id, obj_score in zip(rel_obj_idxs, rel_obj_scores):
                self._index2objid[self._n_queries + self.__n_relevant] = obj_id
                self.__n_relevant += 1
                self._relevant_arr[-1].append(obj_id)
                self._relevance_scores[-1].append(obj_score)

        self._objid2paths = self.__img_paths.groupby("id")[self._input_column].apply(list).to_dict()
        self._objid2paths = {obj_id: self._objid2paths[obj_id] for obj_id in self._index2objid.values()}
        self._data_len = self._n_queries + self.__n_relevant

    def __getitem__(self, index: int) -> dict:
        obj_id = self._index2objid[index]
        image_path = self._objid2paths[obj_id][0]
        image = self._read_image(image_path)
        sample = {"image": image}
        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])
        sample['index'] = index
        return sample

    def __len__(self) -> int:
        return self._data_len

    @property
    def matches(self) -> pd.DataFrame:
        return self.__matches

    @property
    def img_paths(self) -> pd.DataFrame:
        return self.__img_paths

    @property
    def query_arr(self) -> list:
        return self.__query_arr

    @property
    def index2objid(self) -> dict:
        return self._index2objid

    @property
    def objid2paths(self) -> dict:
        return self._objid2paths

    @property
    def n_queries(self) -> int:
        return self._n_queries

    @property
    def data_len(self) -> int:
        return self._data_len
