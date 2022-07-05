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

from src.constructor import METRICS


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
        # in classification the dataset, search_k need to increase by 1, because for every query element first found 
        # retrieval will be itself
        self.search_k = k + 1 if k is not None and self.dataset_type == DatasetType.CLASSIFICATION else k
        # but in metric compute study, k must not change, because in classification search study first retrieval 
        # element would be remove
        self.metric_compute_k = k
        self.add_state('vectors', default=[], dist_reduce_fx=None)
        if self.dataset_type == DatasetType.CLASSIFICATION:
            # if classification dataset
            self.add_state('targets', default=[], dist_reduce_fx=None)
        else:
            # if representation dataset
            self.add_state('query_order_numbers', default=[], dist_reduce_fx=None)
            self.add_state('scores', default=[], dist_reduce_fx=None)

    def update(self, vectors: torch.Tensor, targets: Optional[torch.Tensor] = None,
               query_order_numbers: Optional[torch.Tensor] = None, scores: Optional[torch.Tensor] = None):
        """Append tensors in storage.
        
        Args:
            vectors: Often it would be embeddings, size (batch_size, embedding_size).
            targets: The labels for every vectors in classification mode, size (batch_size).
            query_order_numbers: Integer tensor where values >= 0 represent indices of queries with corresponding 
                vectors in vectors tensor and value -1 indicates that the corresponding vector isn't a query.
            scores: The scores tensor, see representation dataset for more information, 
                size (batch_size, total_num_queries).

        Raises:
            ValueError: If dataset is of classification type and targets is None, or if dataset is of representation 
                type and at least one of scores or queries_idxs is None.
        """
        vectors = vectors.detach().cpu()
        self.vectors.append(vectors)
        if self.dataset_type == DatasetType.CLASSIFICATION:
            if targets is None:
                raise ValueError("In classification dataset target must be not None.")
            targets = targets.detach().cpu()
            self.targets.append(targets)
        else:
            if query_order_numbers is None:
                raise ValueError("In representation dataset query_numbers must be not None.")
            if scores is None:
                raise ValueError("In representation dataset scores must be not None")
            
            query_order_numbers = query_order_numbers.detach().cpu()
            self.query_order_numbers.append(query_order_numbers)
            
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
            relevants, gallery_idxs, queries_idxs = self.prepare_classification_data(targets)
            clear_query_order_numbers = queries_idxs
            clear_scores = None
        else:
            # if representation dataset
            scores = torch.cat(self.scores).numpy()
            query_numbers = torch.cat(self.query_order_numbers).numpy()
            # prepare data
            relevants, clear_scores, gallery_idxs, \
                queries_idxs, clear_query_order_numbers = self.prepare_representation_data(query_numbers, scores)

        # build index
        vectors = vectors.astype(np.float32)
        index = self.build_index(vectors[gallery_idxs])

        # if k is None set it as gallery_idxs length
        search_k = len(gallery_idxs) if self.search_k is None else self.search_k

        # create relevant, closest generator
        generator = self.query_generator(index, vectors, relevants, queries_idxs, 
                                         clear_query_order_numbers, search_k, clear_scores)
        
        # compute metric
        metrics = []
        for relevant_idx, closest_idx in generator:
            metrics += self.metric_func(relevant_idx, closest_idx, k=self.metric_compute_k).tolist()
        metric = np.mean(metrics)
        return metric

    def prepare_representation_data(self, 
                                    query_order_numbers: np.ndarray, 
                                    scores: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in representation dataset case.
        
        Separate query and database vectors from storage vectors.
        Prepare scores.
        Generate relevant indexes for every query request.

        Args:
            query_order_numbers: Integer array where values >= 0 represent indices of queries with corresponding 
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
            relevants: Array of relevant indexes in gallery data, for every query.
            clear_scores: Array of scores with dropped query rows (rows with only zeros values).
            gallery_idxs: Array of gallery indexes in vectors storage.
            queries_idxs: Array of queries indexes in vectors storage.
            clear_query_order_numbers: Array of queries order numbers without -1 elements. It size like queries_idxs,
                so if you want to get order number of queries_idxs[i] it would be clear_query_order_numbers[i].

        Raises:
            ValueError: If dataset has query vector without relevants.
        """
        is_queries = query_order_numbers >= 0
        full_indexes = np.arange(len(query_order_numbers))
        clear_query_order_numbers = query_order_numbers[is_queries]
        queries_idxs = full_indexes[is_queries]
        gallery_idxs = full_indexes[~is_queries]
        clear_scores = scores[~is_queries]

        relevants = []
        # with cycle run through query_order_numbers which corresponding with queries_idxs, so the relevants array
        # would be corresponding with queries_idxs. As a result we will have 3 arrays: 
        # clear_query_order_numbers, queries_idxs and relevants with the same sizes, and for any index i
        # we will have 
        # it's order_number = clear_query_order_numbers[i] in score matrix, 
        # it's index in vectors = queries_idxs[i]
        # it's relevants indexes in gallery data = relevants[i]
        for query_number in clear_query_order_numbers:
            relevant_idxs = np.where(clear_scores[:, query_number] > 0.)[0]
            if len(relevant_idxs) == 0:
                raise ValueError('Representation metric. The dataset contains a query vector that does not '
                                 'has relevants.')
            # Need to sort relevant indexes by its scores for NDCG metric
            current_scores = clear_scores[relevant_idxs, query_number]
            sort_indexes = np.argsort(current_scores)
            relevant_idxs = relevant_idxs[sort_indexes[::-1]]
            relevants.append(relevant_idxs)
        
        relevants = np.array(relevants)
        return relevants, clear_scores, gallery_idxs, queries_idxs, clear_query_order_numbers

    def prepare_classification_data(self, 
                                    targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in classification dataset case.
        
        In the classification case, ll the vectors would be used as queries, and the relevants vectors will be vectors 
        that have the same label as the query.

        Args:
            targets: Targets in classification task for every vector.

        Returns:
            relevants: Array of relevant indexes in gallery data, for every query.
            gallery_idxs: Array of gallery indexes in vectors storage.
            queries_idxs: Array of queries indexes in vectors storage.

        Raises:
            ValueError: If any class has only one element.
        """
        ts = pd.Series(targets)
        groups = pd.Series(ts.groupby(ts).groups)
        # now groups contain group indexes with one targets
        relevants = []
        queries_idxs = []
        # create queries with its relevants indexes
        for i, group in enumerate(groups):
            for query_idx in group:
                # need to drop from relevants index which equal query index
                relevant = group.drop(query_idx).tolist()
                if len(relevant) == 0:
                    raise ValueError(f'Representation metric. The class {groups.index[i]} has only one element.')
                queries_idxs.append(query_idx)
                relevants.append(relevant)

        relevants = np.array(relevants)
        queries_idxs = np.array(queries_idxs)
        gallery_idxs = np.arange(len(targets))
        
        return relevants, gallery_idxs, queries_idxs

    def query_generator(self, index: Union[faiss.swigfaiss_avx2.IndexFlatIP, faiss.swigfaiss_avx2.IndexFlatL2],
                        vectors: np.ndarray, relevants: np.ndarray,
                        queries_idxs: np.ndarray, clear_query_order_numbers: np.ndarray,
                        k: int, clear_scores: Optional[np.ndarray] = None
                        ) -> Generator[Tuple[List[np.ndarray], List[np.ndarray]], None, None]:
        """Create relevants and closest arrays, by faiss index search.

        Output in relevant array, contain it's index in gallery data and score for current query.
        Output in closest array, contain it's index in gallery data and distance = 1 for current query.
        
        This function use self.search_batch_size to define how many vectors to send per one faiss search request.

        Args:
            index: Faiss database built index.
            vectors: All embedding vectors, it contains gallery and queries vectors.
            relevants:  Array of relevant indexes in gallery data, for every query.
            queries_idxs: Array of queries indexes in vectors storage.
            clear_scores: Array of scores with dropped query rows (rows with only zeros values).
            k:  Number of top closest indexes to get.

        Returns:
            Generator which contain relevant and closest Tuple values. 

            Relevant include relevant indexes and scores, size (search_batch_size, , 2).
            Closest include retrieved indexes and scores, size (search_batch_size, , 2).
        """
        for i in range(0, len(queries_idxs), self.search_batch_size):
            batch_idxs = np.arange(i, min(i + self.search_batch_size, len(queries_idxs)))
            # queries_idxs - indexes of vectors (global indexes), so queries vectors can be obtained 
            # like vectors[queries_idxs[batch_idxs]]
            closest_dist, closest_idx = index.search(vectors[queries_idxs[batch_idxs]], k=k)
            
            # remove first element which is actually is it's classification dataset
            if self.dataset_type == DatasetType.CLASSIFICATION:
                closest_dist = np.delete(closest_dist, 0, axis=1)
                closest_idx = np.delete(closest_idx, 0, axis=1)

            # relevants contains gallery indexes - local indexes, so the relevants elemets can be obtained 
            # like relevants[batch_idxs]
            batch_relevant = relevants[batch_idxs]

            if self.metric_distance == MetricDistance.IP:
                closest_dist = 1 - closest_dist

            # NDCG score=distance is needed to sort more relevant examples, but in this part of code we had 
            # already sorted our examples by faiss. So if we change score = 1 to distance with type float 
            # the index of relevant will be also float and after that inside ranx it may be fail to compare 
            # relevant int index with our relevant float index.
            batch_closest = map(lambda idx: 
                                np.stack((closest_idx[idx], [1] * len(closest_idx[idx])), axis=1), 
                                np.arange(len(closest_idx)))

            if clear_scores is None:
                batch_relevant = map(lambda r: np.stack((r, np.ones_like(r)), axis=1), batch_relevant)
            else:
                # clear_query_order_numbers cleared of queries, so it has local indexes and the elements can be 
                # obtained like clear_query_order_numbers[batch_idxs]
                batch_relevant = map(lambda r_q: 
                                     np.stack((r_q[0], clear_scores[r_q[0], clear_query_order_numbers[r_q[1]]]), 
                                              axis=1), 
                                     zip(batch_relevant, batch_idxs))
            
            batch_relevant = list(batch_relevant)
            batch_closest = list(batch_closest)
            yield batch_relevant, batch_closest

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
    