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


class DatasetType(Enum):
    CLASSIFICATION = 0
    REPRESENTATION = 1


class MetricDistance(Enum):
    IP = 0
    L2 = 1


class IndexBasedMeter(Metric, ABC):
    """Base class for representation metrics.

    Store retrieval vectors and targets during phase in update method. FAISS library is used to build an index 
    and search for top-k in it. Supports 2 datasets: classification dataset with targets, 
    and representation dataset with scores and queries_idxs tensors.
    Compute method return generator with relevant and closest (FAISS searched) indexes. The relevant index
    contain it's relevant index with scores for current query. And the closest contain closest index with it's distance.
    """
    def __init__(self, exact_index: bool, dataset_type: DatasetType, metric_distance: MetricDistance, \
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
        self.dataset_type = dataset_type
        self.metric_distance = metric_distance
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
            self.add_state('is_queries', default=[], dist_reduce_fx=None)
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

    def compute(self):
        """Build generator with relevant, closest arrays.
        
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
            is_queries = torch.cat(self.is_queries).numpy()
            # prepare data
            q_vecs, db_vecs, relevants, scores, \
                db_idxs, q_order_idxs = self.prepare_representation_data(vectors, queries_idxs, scores)

        q_vecs = q_vecs.astype(np.float32)
        db_vecs = db_vecs.astype(np.float32)

        q_vecs = q_vecs.astype(np.float32)
        db_vecs = db_vecs.astype(np.float32)

        index = self.build_index(db_vecs)

        # if search batch size is None, search queries vectors by one request
        search_batch_size = len(q_vecs) if self.search_batch_size is None else self.search_batch_size

        # if k is None set it as database length
        k = len(db_vecs) if self.k is None else self.k

        generator = self.query_generator(index, relevants, q_vecs, scores, db_idxs, search_batch_size, k)
        return generator

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
            relevant_idxs = np.where(scores[:, idx] > 0.)[0]
            if len(relevant_idxs) == 0:
                empty_relevant_idxs.append(idx)
            else:
                relevant.append(relevant_idxs)
        relevant = np.array(relevant)

        # remove empty relevant queries
        q_vecs = np.delete(q_vecs, empty_relevant_idxs, axis=0)

        return q_vecs, db_vecs, relevant, scores, db_idxs

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

    def query_generator(self, index: Union[faiss.swigfaiss_avx2.IndexFlatIP, faiss.swigfaiss_avx2.IndexFlatL2], \
                        relevants: np.ndarray, queries: np.ndarray, scores: np.ndarray, db_ids: np.ndarray, \
                        search_batch_size: int, k: int):
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
            for i in range(0, len(queries), search_batch_size):
                if i + search_batch_size >= len(queries):
                    query_idxs = np.arange(i, len(queries))
                else:
                    query_idxs = np.arange(i, i + search_batch_size)

                closest_dist, closest_idx = index.search(queries[query_idxs], k=k)
                relevant = relevants[query_idxs]
 
                if self.metric_distance == MetricDistance.IP:
                    closest_dist = 1 - closest_dist
            
                closest = map(lambda idx: np.stack((db_ids[closest_idx[idx]], closest_dist[idx]), axis=1), \
                    np.arange(len(closest_idx)))
                
                if scores is None:
                    relevant = map(lambda r: np.stack((r, np.ones_like(r)), axis=1), relevant)
                else:
                    relevant = map(lambda r_q: np.stack((r_q[0], scores[r_q[0], r_q[1]]), axis=1), \
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


# @METRICS.register_class
class PrecisionAtKMeter(IndexBasedMeter):
    def compute(self):
        scores = []
        generator = super().compute()
        for relevant_idx, closest_idx in generator:
            scores += precision(relevant_idx, closest_idx, k=self.k).tolist()
        return np.mean(scores)


# @METRICS.register_class
class RecallAtKMeter(IndexBasedMeter):
    def compute(self):
        scores = []
        generator = super().compute()
        for relevant_idx, closest_idx in generator:
            scores += recall(relevant_idx, closest_idx, k=self.k).tolist()
        return np.mean(scores)


# @METRICS.register_class
class MeanAveragePrecisionAtKMeter(IndexBasedMeter):
    def compute(self):
        scores = []
        generator = super().compute()
        for relevant_idx, closest_idx in generator:
            scores += average_precision(relevant_idx, closest_idx, k=self.k).tolist()
        return np.mean(scores)


# @METRICS.register_class
class NDCGAtKMeter(IndexBasedMeter):
    def compute(self):
        scores = []
        generator = super().compute()
        for relevant_idx, closest_idx in generator:
            scores += ndcg(relevant_idx, closest_idx, k=self.k).tolist()
        return np.mean(scores)
    