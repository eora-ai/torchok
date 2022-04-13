from pathlib import Path
from typing import Union, List, Dict, Optional

import numpy as np
import pandas as pd
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose

from .each_to_each_retrieval_dataset import EachToEachRetrievalDataset


class DbRetrievalDataset(EachToEachRetrievalDataset):
    """Dataset for full database image retrieval.
    It's treated that queries and relevant items are separate from the database items

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

    .. csv-table:: DB Image csv example
     :header: id,image_paths
     8,data/db/img_1.jpg
     10,data/db/img_2.jpg
     12,data/db/img_3.jpg

    """
    def __init__(self,
                 data_folder: str,
                 matches_csv_path: str,
                 db_folder: str,
                 include_only_path: str,
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
            db_folder: path to a folder with all database images (traversed recursively)
            include_only_path: sometext
            img_paths_csv_path: path to mapping image identifiers to image paths. Format: id | path.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of of the torch tensors related to the image.
            input_column: Name of the column that contains paths to images.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
        """
        super().__init__(data_folder, matches_csv_path, img_paths_csv_path,
                         transform, augment, input_dtype, input_column, grayscale)
        self.__db_folder = Path(db_folder)

        self.__include_only = pd.read_csv(Path(db_folder) / include_only_path, usecols=["id", input_column],
                                          dtype={"id": str, input_column: str}, header=0)

        self.__include_only[input_column] = self.__include_only[input_column].map(lambda path: self.__db_folder / path)
        self.__db_objid2paths = self.__include_only.groupby('id')[input_column].apply(list).to_dict()
        self._objid2paths.update(self.__db_objid2paths)

        self.__n_db = 0
        for obj_id in self.__db_objid2paths.keys():
            self._index2objid[self._data_len + self.__n_db] = obj_id
            self.__n_db += 1

        self._data_len += self.__n_db
        self.__objid2index = {}

        for index, obj_id in self._index2objid.items():
            if obj_id not in self.__objid2index:
                self.__objid2index[obj_id] = index

        self.__relevance_map = self.__get_relevance_scores()
        self.__targets = np.zeros((len(self), 1 + self._n_queries), dtype=np.float32)

        for index in range(self._n_queries):
            relevant_idxs = self.__relevance_map[index]['relevant_idxs']
            relevant_scores = self.__relevance_map[index]['relevant_scores']
            self.__targets[relevant_idxs, 1 + index] = relevant_scores
            self.__targets[index, 0] = 1.

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        sample['target'] = self.__targets[index]

        return sample

    def __get_relevance_scores(self) -> List[Dict[str, np.ndarray]]:
        """
        Traverses all the query->relevant lists and constructs a map of relevance scores and their positions
        in the joined set of relevant items to each query.

        Returns:
            List of relevance cards to each of N queries.
        Raises:
            ValueError: If relevant objects list doesn't match with relevance scores list in size.
        """
        relevance_map = []

        for index in range(self._n_queries):
            relevant_obj_idxs = self._relevant_arr[index]
            relevance_scores = self._relevance_scores[index]
            if len(relevant_obj_idxs) != len(relevance_scores):
                raise ValueError(f"Relevant objects list must match relevance scores list in size."
                                 f"Got number of relevant object indices: {len(relevant_obj_idxs)}, "
                                 f"number of relevance scores: {len(relevance_scores)}")

            relevant_indices = np.array([self.__objid2index[obj_id] for obj_id in relevant_obj_idxs])
            relevance_map.append({
                'relevant_idxs': relevant_indices,
                'relevant_scores': relevance_scores
            })

        return relevance_map

    @property
    def relevance_map(self) -> List[Dict[str, np.ndarray]]:
        return self.__relevance_map

    @property
    def targets(self) -> np.ndarray:
        return self.__targets

    @property
    def objid2index(self) -> dict:
        return self.__objid2index

    @property
    def db_folder(self) -> Path:
        return self.__db_folder

    @property
    def db_objid2paths(self) -> dict:
        return self.__db_objid2paths

    @property
    def include_only(self) -> pd.DataFrame:
        return self.__include_only
