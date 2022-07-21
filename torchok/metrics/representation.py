from abc import ABC, abstractmethod

from sklearn.preprocessing import normalize
from typing import Callable, List, Optional, Union, Dict, Tuple, Generator

import torch
import faiss
import pandas as pd
import numpy as np
import math

from torchmetrics import Metric
from ranx.metrics import precision, recall, average_precision, ndcg
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
    def __init__(self, exact_index: bool, dataset_type: str, metric_distance: str, \
        metric_func: Callable, k: Optional[int] = None, search_batch_size: Optional[int] = None, \
        normalize_vectors: bool = False, **kwargs):
        """Initialize IndexBasedMeter.

        Args:
            exact_index: If true then build fair inner product or Euclidean index (depends on the metric chosen), 
                otherwise the index will be an approximate nearest neighbours search index.
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
        super().__init__(compute_on_step=False, **kwargs)
        self.exact_index = exact_index
        
        self.dataset_type = dataset_enum_mapping[dataset_type]
        self.metric_distance = distance_enum_mapping[metric_distance]
        self.metric_func = metric_func
        self.normalize_vectors = normalize_vectors
        # set search_batch_size as num CPUs if search_batch_size is None
        self.search_batch_size = torch.get_num_threads() if search_batch_size is None else search_batch_size
        self.k = k

        self.add_state('vectors', default=[], dist_reduce_fx=None)
        if self.dataset_type == DatasetType.CLASSIFICATION:
            # if classification dataset
            self.add_state('targets', default=[], dist_reduce_fx=None)
        else:
            # if representation dataset
            self.add_state('queries_idxs', default=[], dist_reduce_fx=None)
            self.add_state('scores', default=[], dist_reduce_fx=None)

    def update(self, vectors: torch.Tensor, targets: Optional[torch.Tensor] = None, \
               queries_idxs: Optional[torch.Tensor] = None, scores: Optional[torch.Tensor] = None):
        """Append tensors in storage.
        
        Args:
            vectors: Often it would be embeddings, size (batch_size, embedding_size).
            targets: The labels for every vectors in classification mode, size (batch_size).
            queries_idxs: Integer tensor where values >= 0 represent indices of queries with corresponding vectors 
                in vectors tensor and value -1 indicates that the corresponding vector isn't a query.
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
            if queries_idxs is None:
                raise ValueError("In representation dataset queries_idxs must be not None.")
            if scores is None:
                raise ValueError("In representation dataset scores must be not None")
            
            queries_idxs = queries_idxs.detach().cpu()
            self.queries_idxs.append(queries_idxs)
            
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
            Metric value.
        """
        vectors = torch.cat(self.vectors).numpy()
        
        if self.normalize_vectors:
            vectors = normalize(vectors)

        if self.dataset_type == DatasetType.CLASSIFICATION:
            # if classification dataset
            targets = torch.cat(self.targets).numpy()
            # prepare data
            q_vecs, db_vecs, relevants, scores, \
                db_idxs, q_order_idxs = self.prepare_classification_data(vectors, targets)
        else:
            # if representation dataset
            scores = torch.cat(self.scores).numpy()
            queries_idxs = torch.cat(self.queries_idxs).numpy()
            # prepare data
            q_vecs, db_vecs, relevants, scores, \
                db_idxs, q_order_idxs = self.prepare_representation_data(vectors, queries_idxs, scores)

        q_vecs = q_vecs.astype(np.float32)
        db_vecs = db_vecs.astype(np.float32)

        index = self.build_index(db_vecs)

        # if k is None set it as database length
        k = len(db_vecs) if self.k is None else self.k

        # create relevant, closest generator
        generator = self.query_generator(index, relevants, q_vecs, db_idxs, q_order_idxs, k, scores)
        
        # compute metric
        scores = []
        for relevant_idx, closest_idx in generator:
            scores += self.metric_func(relevant_idx, closest_idx, k=k).tolist()
        return np.mean(scores)

    def prepare_representation_data(self, vectors: np.ndarray, queries_idxs: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in representation dataset case.
        
        Separate query and database vectors from storage vectors.
        Prepare scores.
        Generate relevant indexes for every query request.

        Args:
            vectors: Vectors of the whole storage, includes queries and database vectors, 
                size (dataset_size, embedding_size).
            queries_idxs: Integer tensor where values >= 0 represent indices of queries with corresponding vectors 
                in vectors tensor and value -1 indicates that the corresponding vector isn't a query.
            scores: The scores tensor, see representation dataset for more information, 
                size (batch_size, total_num_queries).

        Returns:
            q_vecs: Queries vectors, size (queries_size, embedding_size).
            db_vecs: Database vectors, for faiss build index, size (database_size, embedding_size).
            relevant: Array of arrays relevant indexes in database for every query request, size (queries_size, ).
            scores: Array of scores related to queries that have at least one relevant item.
            db_idxs: Array with all database indexes.
            queries_idxs: Array of queries order number.
        """
        is_queries = queries_idxs >= 0
        queries_idxs = queries_idxs[is_queries]
        q_vecs = vectors[is_queries]
        db_vecs = vectors[~is_queries]
        db_idxs = np.arange(len(db_vecs))
        scores = scores[~is_queries]
        
        relevant = []
        empty_relevant_idxs = []

        for idx in range(len(q_vecs)):
            relevant_idxs = np.where(scores[:, queries_idxs[idx]] > 0.)[0]
            if len(relevant_idxs) == 0:
                empty_relevant_idxs.append(idx)
            else:
                # Need to sort relevant indexes by its scores for NDCG metric
                current_scores = scores[relevant_idxs, queries_idxs[idx]]
                sort_indexes = np.argsort(current_scores)
                relevant_idxs = relevant_idxs[sort_indexes[::-1]]
                relevant.append(relevant_idxs)
        
        relevant = np.array(relevant)

        # remove empty relevant queries
        q_vecs = np.delete(q_vecs, empty_relevant_idxs, axis=0)
        queries_idxs = np.delete(queries_idxs, empty_relevant_idxs)
        
        return q_vecs, db_vecs, relevant, scores, db_idxs, queries_idxs

    def prepare_classification_data(self, vectors: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in classification dataset case.
        
        Separate query and database vectors from storage vectors.
        Query vector index will be the first row index for each unique target value found in targets array. 
        So, total number of queries is equal to unique target count. 
        The remaining vectors will be treated as database vectors (again for each unique target value)

        Args:
            vectors: Vectors of the whole storage, includes queries and database vectors, 
                size (dataset_size, embedding_size).
            targets: Targets in classification task for every vector, size (database_size).

        Retruns:
            q_vecs: Queries vectors, size (queries_size, embedding_size).
            db_vecs: Database vectors, for faiss build index, size (database_size, embedding_size).
            relevant: Array of relevant indexes in database for every query request, size (queries_size, ).
            scores: Array of scores without queries empty scores.
            db_idxs: Array with all database indexes.
            queries_idxs: Array of queries order number.
        """
        ts = pd.Series(targets)

        # Group item indices of the same class
        groups = pd.Series(ts.groupby(ts).groups)

        # Take the first index of the group as a query
        # If there is only one element in the group, ignore this group
        query_idxs = groups.apply(lambda x: x[0] if len(x) > 1 else -1)

        # Rest indices are used as a relevant/database elements
        relevant = groups.apply(lambda x: x[1:].values if len(x) > 1 else x.values)

        # Create mapping from database index to original index
        db_idxs = np.array(sorted([i for j in relevant for i in j]))

        # Filter groups with more than one sample per group
        correct_classes = query_idxs != -1
        relevant = relevant[correct_classes]
        query_idxs = query_idxs[correct_classes]

        # Retrieve query and database vectors and build index base on latter.
        q_vecs = vectors[query_idxs]
        db_vecs = vectors[db_idxs]

        scores = None
        query_order_idxs = np.arange(len(q_vecs))
        return q_vecs, db_vecs, relevant, scores, db_idxs, query_order_idxs

    def query_generator(self, index: Union[faiss.IndexFlatIP, faiss.IndexFlatL2],
                        relevants: np.ndarray, queries: np.ndarray, db_ids: np.ndarray, q_order_idxs: np.ndarray, 
                        k: int, scores: Optional[np.ndarray] = None
    ) -> Generator[Tuple[List[np.ndarray], List[np.ndarray]], None, None]:
        """Create relevants relevant, closest arrays.

        Output in relevant array, contain it's index in database and score for current query.
        Output in closest array, contain it's index in database and distance for current query.
        
        Args:
            index: Faiss database built index.
            relevants: Relevant indexes for every query, size (total_num_queries, ) and the second shape is can be different
                for every query request.
            queries: Vectors for every query request, size (total_num_queries, embedding_size).
            scores: Scores for every relevant index per each query request, size (database_size, total_num_queries).
                See representation dataset for more information.
            db_ids: Database indexes.
            k:  Number of top closest indexes to get.

        Returns:
            Generator wich contain relevant and closest Tuple values. 

            Relevant include relevant indexes and scores, size (search_batch_size, , 2).
            Closest include searched indexes and distances, size (search_batch_size, , 2).
        """
        def generator():
            """Generate relevant - y_true, and closest - y_pred for metric calculation with ranx library.

            Returns:
                relevant: List of relevant indexes with its scores per each queries. Length of list = search_batch_size.
                    And size of one element of list = (relevant_indexes_size, 2), where last shape 2 for relevant index
                    and it score. 
                    Example for 3 search_batch_size, and relevant_sizes = [2, 2, 1] with score = 1 for every \
                    relevant index:
                    [
                        np.array([[6, 1], [7, 1]]), 
                        np.array([[2, 1], [5, 1]]), 
                        np.array([[4, 1]])
                    ].
                closest: List of numpy arrays, with nearest searched indexes by top k.
                    Example for k = 3:
                    [
                        np.array([[4, 1],
                               [2, 1],
                               [6, 1]]), 
                        np.array([[5, 1],
                               [2, 1],
                               [6, 1]]), 
                        np.array([[4, 1],
                               [5, 1],
                               [6, 1]])
                    ].
            """
            for i in range(0, len(queries), self.search_batch_size):
                query_idxs = np.arange(i, min(i + self.search_batch_size, len(queries)))

                closest_dist, closest_idx = index.search(queries[query_idxs], k=k)
                relevant = relevants[query_idxs]
 
                if self.metric_distance == MetricDistance.IP:
                    closest_dist = 1 - closest_dist

                # NDCG score=distance is needed to sort more relevant examples, but in this part of code we had 
                # already sorted our examples by faiss. So if we change score = 1 to distance with type float 
                # the index of relevant will be also float and after that inside ranx it may be fail to compare 
                # relevant int index with our relevant float index.
                closest = map(lambda idx: np.stack((db_ids[closest_idx[idx]], [1] * len(closest_idx[idx])), axis=1), \
                    np.arange(len(closest_idx)))
                
                if scores is None:
                    relevant = map(lambda r: np.stack((r, np.ones_like(r)), axis=1), relevant)
                else:
                    relevant = map(lambda r_q: np.stack((r_q[0], scores[r_q[0], q_order_idxs[r_q[1]]]), axis=1), \
                        zip(relevant, query_idxs))
                
                relevant = list(relevant)
                closest = list(closest)

                yield relevant, closest

        return generator()

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
class PrecisionAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True, \
                 metric_distance: str = 'IP', k: Optional[int] = None, \
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance, \
            metric_func=precision, k=k, search_batch_size=search_batch_size, normalize_vectors=normalize_vectors, \
            **kwargs)


@METRICS.register_class
class RecallAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True, \
                 metric_distance: str = 'IP', k: Optional[int] = None, \
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance, \
            metric_func=recall, k=k, search_batch_size=search_batch_size, normalize_vectors=normalize_vectors, \
            **kwargs)


@METRICS.register_class
class MeanAveragePrecisionAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True, \
                 metric_distance: str = 'IP', k: Optional[int] = None, \
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance, \
            metric_func=average_precision, k=k, search_batch_size=search_batch_size, \
            normalize_vectors=normalize_vectors, **kwargs)


@METRICS.register_class
class NDCGAtKMeter(IndexBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True, \
                 metric_distance: str = 'IP', k: Optional[int] = None, \
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance, \
            metric_func=ndcg, k=k, search_batch_size=search_batch_size, normalize_vectors=normalize_vectors, \
            **kwargs)
    