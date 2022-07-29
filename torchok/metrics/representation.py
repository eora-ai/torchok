from abc import ABC

from sklearn.preprocessing import normalize
from typing import Callable, List, Optional, Union, Tuple, Generator

import torch
import faiss
import numpy as np
import pandas as pd
import math

from torchmetrics import Metric
from ranx.metrics import precision, recall, average_precision, ndcg, hit_rate
from enum import Enum

from torchok.constructor import METRICS


__all__ = [
    'PrecisionAtKMeter',
    'RecallAtKMeter',
    'MeanAveragePrecisionAtKMeter',
    'NDCGAtKMeter',
]


class DatasetType(Enum):
    CLASSIFICATION = 'classification'
    REPRESENTATION = 'representation'


class MetricDistance(Enum):
    IP = 'IP'
    L2 = 'L2'


dataset_enum_mapping = {
    'classification': DatasetType.CLASSIFICATION,
    'representation': DatasetType.REPRESENTATION,
}

distance_enum_mapping = {
    'IP': MetricDistance.IP,
    'L2': MetricDistance.L2
}


class IndexBasedMeter(Metric, ABC):
    """Base class for representation metrics.

    Store retrieval vectors and targets during phase in update method. FAISS library is used to build an index
    and search for top-k in it. Supports 2 datasets: classification dataset with targets,
    and representation dataset with scores and queries_idxs tensors.
    Compute method return generator with relevant and closest (FAISS searched) indexes. The relevant index
    contain it's relevant index with scores for current query. And the closest contain closest index with it's distance.
    """
    def __init__(self, exact_index: bool, dataset_type: str, metric_distance: str,
                 metric_func: Callable, k: Optional[int] = None, search_batch_size: Optional[int] = None,
                 normalize_vectors: bool = False, **kwargs):
        """Initialize IndexBasedMeter.

        Args:
            exact_index: If true then build fair inner product or Euclidean index (depends on the metric chosen),
                otherwise the index will be an approximate nearest neighbors search index.
            dataset_type: Dataset type (classification or representation), which will be used to calculate metric.
            metric_distance: Metric distance (IP - cosine distance, L2 - euclidean), which will be used to build
                FAISS index.
            metric_func: Representation metric (e.g ranx metric function) with the follow backend
                `def metric_func(
                    qrels: Union[np.ndarray, numba.typed.List],
                    run: Union[np.ndarray, numba.typed.List],
                    k: int = 0,
                ) -> np.ndarray`
                where qrels - y_true and run - y_pred
                see https://github.com/AmenRa/ranx/blob/ccab1549de81e7366e34213c86089e965db55f72/ranx/metrics.py
                for more details.
            k: Number of top closest indexes to get.
            search_batch_size: The size for one FAISS search request.
            normalize_vectors: If true vectors will be normalize, overwise no.

        Raises:
            ValueError: If metric or dataset is not correct write.
        """
        super().__init__(**kwargs)
        self.exact_index = exact_index

        self.dataset_type = dataset_enum_mapping[dataset_type]
        self.metric_distance = distance_enum_mapping[metric_distance]
        self.metric_func = metric_func
        self.normalize_vectors = normalize_vectors
        # set search_batch_size as num CPUs if search_batch_size is None
        self.search_batch_size = torch.get_num_threads() if search_batch_size is None else search_batch_size

        # set k as 1 if k is None
        k = 1 if k is None else k
        # as queries vectors can be relevant for another query vectors, so we should search k + 1 and delete first
        # and remove the first found vector because it would be query
        self.search_k = k + 1
        # but in metric compute study, k must not change
        self.metric_compute_k = k

        self.add_state('vectors', default=[], dist_reduce_fx=None)
        if self.dataset_type == DatasetType.CLASSIFICATION:
            # if classification dataset
            self.add_state('targets', default=[], dist_reduce_fx=None)
        else:
            # if representation dataset
            self.add_state('query_idxs', default=[], dist_reduce_fx=None)
            self.add_state('scores', default=[], dist_reduce_fx=None)

    def update(self, vectors: torch.Tensor, targets: Optional[torch.Tensor] = None,
               query_idxs: Optional[torch.Tensor] = None, scores: Optional[torch.Tensor] = None):
        """Append tensors in storage.

        Args:
            vectors: Often it would be embeddings, size (batch_size, embedding_size).
            targets: The labels for every vectors in classification mode, size (batch_size).
            query_idxs: Integer tensor where values >= 0 represent indices of queries with corresponding
                vectors in vectors tensor and value -1 indicates that the corresponding vector isn't a query.
            scores: The scores tensor, see representation dataset for more information,
                size (batch_size, total_num_queries).

        Raises:
            ValueError: If dataset is of classification type and targets is None, or if dataset is of representation
                type and at least one of scores or query_idxs is None.
        """
        vectors = vectors.detach().cpu()
        self.vectors.append(vectors)
        if self.dataset_type == DatasetType.CLASSIFICATION:
            if targets is None:
                raise ValueError("In classification dataset target must be not None.")
            targets = targets.detach().cpu()
            self.targets.append(targets)
        else:
            if query_idxs is None:
                raise ValueError("In representation dataset query_numbers must be not None.")
            if scores is None:
                raise ValueError("In representation dataset scores must be not None")

            query_idxs = query_idxs.detach().cpu()
            self.query_idxs.append(query_idxs)

            scores = scores.detach().cpu()
            self.scores.append(scores)

    def compute(self) -> float:
        """Compute metric value.

        Firstly it gathers all tensors in storage (done by torchmetrics).
        Then it prepares data, separates query and database vectors.
        Then it builds the FAISS index.
        Then, it create a generator of relevant and closest arrays.
        Finally, it compute metric.

        Returns:
            metric: Metric value.
        """
        vectors = torch.cat(self.vectors).numpy()
        if self.normalize_vectors:
            vectors = normalize(vectors)

        if self.dataset_type == DatasetType.CLASSIFICATION:
            # if classification dataset
            targets = torch.cat(self.targets).numpy()
            # prepare data
            relevant_idxs, gallery_idxs, query_row_idxs, query_as_relevant = self.prepare_classification_data(targets)
            # mock scores and query column indexes because it belong to representation data
            scores = None
            query_column_idxs = None
        else:
            # if representation dataset
            scores = torch.cat(self.scores).numpy()
            query_idxs = torch.cat(self.query_idxs).numpy()
            # prepare data
            relevant_idxs, gallery_idxs, query_column_idxs, \
                query_row_idxs, query_as_relevant = self.prepare_representation_data(query_idxs, scores)

        # build index
        vectors = vectors.astype(np.float32)
        index = self.build_index(vectors[gallery_idxs])

        # create relevant, closest generator
        generator = self.query_generator(index, vectors, relevant_idxs, query_row_idxs,
                                         gallery_idxs, query_as_relevant, self.search_k, scores, query_column_idxs)

        # compute metric
        metrics = []
        for relevant_idx, closest_idx in generator:
            metrics += self.metric_func(relevant_idx, closest_idx, k=self.metric_compute_k).tolist()
        metric = np.mean(metrics)
        return metric

    def prepare_representation_data(self,
                                    query_idxs: np.ndarray,
                                    scores: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in representation dataset case.

        Separate query and database vectors from storage vectors.
        Prepare scores.
        Generate relevant indexes for every query request.

        Args:
            query_idxs: Integer array where values >= 0 represent indices of queries with corresponding
                vectors in vectors tensor and value -1 indicates that the corresponding vector isn't a query. Also
                value of this array is the column number in the scores matrix, which need to reproduce
                relevant elements.
            scores: The scores tensor, see representation dataset for more information,
                size (batch_size, total_num_queries).
                Example of score matrix:
                    SCORES = torch.tensor(
                        [
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 2, 0],
                            [0, 0, 1],
                            [0, 0, 2],
                            [0, 0, 4],
                            [0, 4, 0],
                        ]
                    ),
                it has 3 queries in [0, 2, 3] row position where only zeros, in order to find relevant vector index you
                need to look at the column, first column for first query, second column for second and so on. For query
                0 relevant vector index would be 1 with score 1, for query 1 - relevant = [4, 8] with scores = [2, 4]
                respectively, for query 2 - relevant = [5, 6, 7] with scores = [1, 2 , 4] respectively.

        Returns:
            relevant_idxs: Array of relevant indexes in gallery data, for every query.
            gallery_idxs: Array of gallery indexes in vectors storage.
            query_column_idxs: Array of queries column indexes in scores matrix. It means query order number in origin
                dataset, because after shuffle dataset the query order may be change we need to store it to reproduce
                for every query vector it's relevant.
            query_row_idxs: Array of queries row indexes in scores matrix, lso this index belong to vectors storage,
                need to get query vector in search study.
            query_as_relevant: Array of query row indexes which are in relevant, i.e belong to queries and
                gallery simultaneously.

        Raises:
            ValueError: If dataset has query vector without relevants.
        """
        is_query = query_idxs >= 0
        # array of columns indexes in scores matrix, need to get relevant
        query_column_idxs = query_idxs[is_query]
        # array of row indexes in scores matrix, also this index belong to vectors storage, need to get query vector in
        # search study
        query_row_idxs = np.where(is_query)[0]
        # gallery idxs in vectors storage, which row sum > 0
        # TODO: try to get gallery indexes from Dataset
        gallery_idxs = np.where(np.any(scores > 0, axis=-1))[0]
        # found query row indexes which are in relevant, i.e belong to queries and gallery simultaneously
        query_as_relevant = np.in1d(query_row_idxs, gallery_idxs)

        relevant_idxs = []
        for query_col_idx in query_column_idxs:
            curr_relevant_idxs = np.where(scores[:, query_col_idx] > 0.)[0]
            if len(curr_relevant_idxs) == 0:
                raise ValueError('Representation metric. The dataset contains a query vector that does not '
                                 'has relevants.')
            # Need to sort relevant indexes by its scores for NDCG metric
            current_scores = scores[curr_relevant_idxs, query_col_idx]
            sort_indexes = np.argsort(current_scores)
            curr_relevant_idxs = curr_relevant_idxs[sort_indexes[::-1]]
            relevant_idxs.append(curr_relevant_idxs)

        relevant_idxs = np.array(relevant_idxs)
        return relevant_idxs, gallery_idxs, query_column_idxs, query_row_idxs, query_as_relevant

    def prepare_classification_data(self,
                                    targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in classification dataset case.

        In the classification case, ll the vectors would be used as queries, and the relevants vectors will be vectors
        that have the same label as the query.

        Args:
            targets: Targets in classification task for every vector.

        Returns:
            relevant_idxs: Array of relevant indexes in gallery data, for every query.
            gallery_idxs: Array of gallery indexes in vectors storage.
            query_row_idxs: Array of queries indexes in vectors storage.
            query_as_relevant: Array of query row indexes which are in relevant, i.e belong to queries and
                gallery simultaneously.
        Raises:
            ValueError: If any class has only one element.
        """
        ts = pd.Series(targets)
        groups = pd.Series(ts.groupby(ts).groups)
        # now groups contain group indexes with one targets
        relevant_idxs = []
        query_row_idxs = []
        # create queries with its relevants indexes
        for i, group in enumerate(groups):
            for query_idx in group:
                # need to drop from relevants index which equal query index
                relevant = group.drop(query_idx).tolist()
                if len(relevant) == 0:
                    raise ValueError(f'Representation metric. The class {groups.index[i]} has only one element.')
                query_row_idxs.append(query_idx)
                relevant_idxs.append(relevant)

        relevant_idxs = np.array(relevant_idxs)
        query_row_idxs = np.array(query_row_idxs)
        gallery_idxs = np.arange(len(targets))
        # all query vectors is in relevant, so create array with True elements
        query_as_relevant = np.full((len(gallery_idxs),), fill_value=True, dtype=np.bool)
        return relevant_idxs, gallery_idxs, query_row_idxs, query_as_relevant

    def query_generator(self, index: Union[faiss.IndexFlatIP, faiss.IndexFlatL2],
                        vectors: np.ndarray, relevants_idxs: np.ndarray,
                        query_row_idxs: np.ndarray, gallery_idxs: np.ndarray, query_as_relevant: np.ndarray,
                        k: int, scores: Optional[np.ndarray] = None, query_col_idxs: Optional[np.ndarray] = None
                        ) -> Generator[Tuple[List[np.ndarray], List[np.ndarray]], None, None]:
        """Create relevants and closest arrays, by faiss index search.

        Output in relevant array, contain it's index in gallery data and score for current query.
        Output in closest array, contain it's index in gallery data and distance = 1 for current query.

        This function use self.search_batch_size to define how many vectors to send per one faiss search request.

        Need to know, that query_row_idxs, query_col_idxs and query_as_relevant - have the same size, and for i-th
        query element the following information is available:
            query_row_idxs[i] - index to get embedding vector in vectors storage
            query_col_idxs[i] - column index in scores matrix, need to get score value for every relevant (used in NDCG)
            query_as_relevant[i] - whether the query in relevant i.e in gallery data, need to remove first element in
                retrieved indexes if it true, and last element if it false (because in fact, k + 1 search request is
                being made)

        Args:
            index: Faiss database built index.
            vectors: All embedding vectors, it contains gallery and queries vectors.
            relevants_idxs: Array of relevant indexes for every query, index belong vectors storage.
            query_row_idxs: Array of query indexes in vectors storage.
            gallery_idxs: Array of gallery indexes in vectors storage.
            query_as_relevant: Boolean array of indexes which indicates if query is in relevant set,
                i.e belong to queries and gallery simultaneously.
            k:  Number of top closest indexes to get.
            scores: Array of scores.
            query_col_idxs: Array of query row indexes which are in relevant, i.e belong to queries and
                gallery simultaneously.

        Returns:
            Generator which contain relevant and closest Tuple values.

            Relevant include relevant indexes and scores, size (search_batch_size, , 2).
            Closest include retrieved indexes and scores, size (search_batch_size, , 2).
        """
        for i in range(0, len(query_row_idxs), self.search_batch_size):
            batch_idxs = np.arange(i, min(i + self.search_batch_size, len(query_row_idxs)))

            batch_query_as_relevant = query_as_relevant[batch_idxs]
            batch_query_row_idxs = query_row_idxs[batch_idxs]
            batch_relevants_idxs = relevants_idxs[batch_idxs]

            # queries_idxs - indexes of vectors (global indexes), so queries vectors can be obtained
            # like vectors[queries_idxs[batch_idxs]]
            _, local_closest_idxs = index.search(vectors[batch_query_row_idxs], k=k)
            # get global indexes
            batch_closest_idxs = gallery_idxs[local_closest_idxs]
            # need delete elements same as query elements in searched indexes
            closest_idxs_delete_firs = np.delete(batch_closest_idxs[batch_query_as_relevant], 0, axis=1)
            # or delete last element because we use k + 1 for search
            closest_delete_last = np.delete(batch_closest_idxs[~batch_query_as_relevant], -1, axis=1)
            # create new matrix from closest_idxs_delete_firs and closest_delete_last
            batch_closest_idxs = np.zeros((len(batch_idxs), k - 1))
            batch_closest_idxs[batch_query_as_relevant] = closest_idxs_delete_firs
            batch_closest_idxs[~batch_query_as_relevant] = closest_delete_last

            # NDCG score=distance is needed to sort more relevant examples, but in this part of code we had
            # already sorted our examples by faiss. So if we change score = 1 to distance with type float
            # the index of relevant will be also float and after that inside ranx it may be fail to compare
            # relevant int index with our relevant float index.
            closest_idxs = map(lambda idx:
                               np.stack((batch_closest_idxs[idx], [1] * len(batch_closest_idxs[idx])), axis=1),
                               np.arange(len(batch_closest_idxs)))

            if scores is None or query_col_idxs is None:
                search_relevants_idxs = map(lambda r: np.stack((r, np.ones_like(r)), axis=1), batch_relevants_idxs)
            else:
                batch_query_col_idxs = query_col_idxs[batch_idxs]
                # clear_query_order_numbers cleared of queries, so it has local indexes and the elements can be
                # obtained like clear_query_order_numbers[batch_idxs]
                search_relevants_idxs = map(lambda r_q:
                                            np.stack((r_q[0], scores[r_q[0], r_q[1]]), axis=1),
                                            zip(batch_relevants_idxs, batch_query_col_idxs))

            search_relevants_idxs = list(search_relevants_idxs)
            closest_idxs = list(closest_idxs)
            yield search_relevants_idxs, closest_idxs

    def build_index(self, vectors: np.ndarray):
        """Build index of a given set of vectors with FAISS.

        Args:
            vectors: Database vectors to index, size (database_size, embedding_size).

        Returns:
            Constructed index.
        """
        vectors = vectors.astype(np.float32)
        n, d = vectors.shape

        index_class = faiss.IndexFlatIP if self.metric_distance == MetricDistance.IP else faiss.IndexFlatL2
        if self.exact_index:
            index = index_class(d)
        else:
            nlist = 4 * math.ceil(n ** 0.5)
            quantiser = index_class(d)
            index = faiss.IndexIVFFlat(quantiser, d, nlist, self.metric_distance.value)
            index.train(vectors)

        index.add(vectors)
        return index


@METRICS.register_class
class HitAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=hit_rate, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, **kwargs)


@METRICS.register_class
class PrecisionAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=precision, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, **kwargs)


@METRICS.register_class
class RecallAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=recall, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, **kwargs)


@METRICS.register_class
class MeanAveragePrecisionAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=average_precision, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, **kwargs)


@METRICS.register_class
class NDCGAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=ndcg, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, **kwargs)
