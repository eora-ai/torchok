from abc import ABC, abstractmethod

from sklearn.preprocessing import normalize
from typing import List, Optional, Union, Dict, Tuple

import torch
from torchmetrics import Metric
from ranx.metrics import precision, recall, average_precision, ndcg
import faiss
import pandas as pd
import numpy as np
import math


class IndexBasedMeter(Metric, ABC):
    """Base class for representation metrics.

    Store some vectors and over tensors over phase in update method.
    Support 2 datasets: classification dataset with targets, and representation dataset with scores and is_query
        tensors.
    Using faiss library build index and search top k.
    Compute method return generator with relevant and closest (faiss searched) indexes. The relevant index
    contain it's relevant index with scores for current query. And the closest contain closest index with it's distance.

    Args:
        exact_index: If true compute clear faiss index, overwise would be approximation.
        dataset: The dataset name that would be use to calculate the metric.
        metric: Faiss used distance type.
        k: Number of top closest indexes to get.
        search_batch_size: The size for one faiss search request.
        normalize_input: If true vectors woulds be normalize, overwise no.

    Raises:
        ValueError: If metric or dataset is not correct write.
    """
    metrics = {'IP': 0, 'L2': 1}
    dataset = {'classification': 0, 'representation': 1}
    def __init__(self, exact_index: bool, dataset: Union[str, int], metric: Union[str, int], k: int = None, \
                 search_batch_size: int = None, normalize_input: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.exact_index = exact_index
        # raise if metric is incorrect
        if not (metric in self.metrics or metric in self.metrics.values()):
            raise ValueError("`metric` must be " + ' | '.join([f"{i}" for j in self.metrics.items() for i in j]))
        # raise if dataset is incorrect
        if not (dataset in self.dataset or dataset in self.dataset.values()):
            raise ValueError("`dataset` must be " + ' | '.join([f"{i}" for j in self.dataset.items() for i in j]))
        
        self.dataset = self.dataset.get(dataset, dataset)
        self.metric = self.metrics.get(metric, metric)
        self.normalize_input = normalize_input

        self.search_batch_size = search_batch_size
        self.k = k

        self.add_state('vectors', default=[], dist_reduce_fx=None)
        if self.dataset == 0:
            # if classification dataset
            self.add_state('targets', default=[], dist_reduce_fx=None)
        else:
            # if representation dataset
            self.add_state('is_queries', default=[], dist_reduce_fx=None)
            self.add_state('scres', default=[], dist_reduce_fx=None)

    def update(self, vectors: torch.Tensor, targets: Optional[torch.Tensor] = None, \
               is_queries: Optional[torch.Tensor] = None, scores: Optional[torch.Tensor] = None):
        """Append tensors in storage.
        
        Args:
            vectors: Often it would be embeddings, size (batch_size, embedding_size).
            targets: The vectors targets, size (batch_size).
            is_queries: Bool tensor that say if vector in vectors is query, size (batch_size).
            scores: The scores tensor, see representation dataset for more information, 
                size (batch_size, all_query_count).

        Raises:
            ValueError: If dataset is classification and targets is None, or if dataset is representation and one of
                scores or is_queries is None.
        """
        vectors = vectors.detach().cpu()
        self.vectors.append(vectors)
        if self.dataset == 0:
            if targets is None:
                raise ValueError("In classification dataset target must be a not None value.")
            targets = targets.detach().cpu()
            self.targets.append(targets)
        else:
            if is_queries is None:
                raise ValueError("In representation dataset is_queries must be a not None value.")
            if scores is None:
                raise ValueError("In representation dataset scores must be a not None")
            
            is_queries = is_queries.detach().cpu()
            self.is_queries.append(is_queries)
            
            scores = scores.detach().cpu()
            self.scores.append(scores)

    def compute(self) -> function:
        """Build generator with relevant, closest arrays.
        
        Firstly it gather all storage.
        Prepare data, separate queries vectors and data base vectors.
        Then build faiss index.
        Create generator of relevant, closest arrays.

        Returns:
            Generator wich return relevant and closest examples.
        """
        vectors = torch.cat(self.vectors).numpy()
        if self.normalize_input:
            vectors = normalize(vectors)
        if self.dataset == 0:
            # if classification dataset
            targets = torch.cat(self.targets).numpy()
            # prepare data
            q_vecs, db_vecs, relevants, scores, db_idxs = self.prepare_classification_data(vectors, targets)
        else:
            # if representation dataset
            scores = torch.cat(self.scores).numpy()
            is_queries = torch.cat(self.is_quieries).numpy()
            # prepare data
            q_vecs, db_vecs, relevants, scores, db_idxs = self.prepare_representation_data(vectors, is_queries, scores)

        index = self.build_index(db_vecs)

        batch_size = len(q_vecs) if self.search_batch_size is None else self.search_batch_size
        k = len(db_vecs) if self.k is None else self.k

        generator = self.query_generator(self, index, relevants, q_vecs, scores, db_idxs, batch_size, k)
        return generator

    def prepare_representation_data(self, vectors: np.ndarray, is_queries: np.ndarray, scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in representation dataset case.
        
        Separate query and database vectors from storage vectors.
        Prepare scores.
        Generate relevant indexes for every query request.

        Args:
            vectors: Vectors of all storage, include queries and database vectors, size (database_size, embedding_size).
            is_queries: Bool array wich indicate if vector is query, size (database_size).
            scores: Array of scores for every relevant index per each query request, size (database_size, query_size).
                See representation dataset for more information.

        Retruns:
            q_vecs: Queries vectors, size (queries_size, embedding_size).
            db_vecs: Database vectors, for faiss build index, size (database_size, embedding_size).
            relevant: Array of relevant indexes in database for every query request, size (queries_size, ).
            scores: Array of scores without queries empty scores.
            db_idxs: Array with all database indexes.
        """
        q_vecs = vectors[is_queries]
        db_vecs = vectors[~is_queries]
        db_idxs = np.arange(len(db_vecs))
        scores = scores[len(q_vecs):]
        relevant = []
        for idx in range(len(q_vecs)):
            relevant.append(np.where(scores[:, idx] > 0.)[0])
        relevant = np.array(relevant)
        return q_vecs, db_vecs, relevant, scores, db_idxs

    def prepare_classification_data(self, vectors: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in classification dataset case.
        
        Separate query and database vectors from storage vectors.
        Query vectors index would be first found target uniq value in targets array. Query size is uniq target count.

        Args:
            vectors: Vectors of all storage, include queries and database vectors, size (database_size, embedding_size).
            targets: Targets in classification task for every vectors, size (database_size).

        Retruns:
            q_vecs: Queries vectors, size (queries_size, embedding_size).
            db_vecs: Database vectors, for faiss build index, size (database_size, embedding_size).
            relevant: Array of relevant indexes in database for every query request, size (queries_size, ).
            scores: Array of scores without queries empty scores.
            db_idxs: Array with all database indexes.
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
        return q_vecs, db_vecs, relevant, scores, db_idxs

    def query_generator(self, index: Union[faiss.swigfaiss_avx2.IndexFlatIP, faiss.swigfaiss_avx2.IndexFlatL2], \
                        relevants: np.ndarray, queries: np.ndarray, scores: np.ndarray, db_ids: np.ndarray, \
                        search_batch_size: int, k: int) -> function:
        """Create relevants relevant, closest arrays.

        Output in relevant array, contain it's index in database and score for current query.
        Output in closest array, contain it's index in database and distance for current query.
        
        Args:
            index: Faiss database built index.
            relevants: Relevant indexes for every query, size (query_size, ) and the second shape is can be different
                for every query request.
            queries: Vectors for every query request, size (query_size, embedding_size).
            scores: Scores for every relevant index per each query request, size (database_size, query_size).
                See representation dataset for more information.
            db_ids: Database indexes.
            search_batch_size: Query size for one search request.
            k:  Number of top closest indexes to get.

        Returns:
            Generator wich contain relevant and closest Tuple values. 

            Relevant include relevant indexes and scores, size (search_batch_size, , 2).
            Closest include searched indexes and distances, size (search_batch_size, , 2).
        """
        def generator():
            for i in range(len(queries) + search_batch_size):
                if i + search_batch_size >= len(queries):
                    query_idxs = np.arange(i, len(queries))
                else:
                    query_idxs = np.arange(i, i + search_batch_size)
                
                closest_dist, closest_idx = index.search(queries[query_idxs], k=k)
                relevant = relevants[query_idxs]

                if self.metric == 0:
                    closest_dist = 1 - closest_dist

                closest = np.dstack([db_ids[closest_idx], closest_dist])[0]
                if scores is None:
                    relevant = np.stack((relevant, np.ones_like(relevant)), axis=1)
                else:
                    relevant = np.stack((relevant, scores[relevant, query_idxs]), axis=1)

                yield relevant, closest

        return generator()

    def build_index(self, vectors: np.ndarray):
        """Build index with faiss of a given set of vectors.

        Args:
            vectors: Database vectors on with is built faiss index, size (database_size, embedding_size).

        Returns:
            Constructed index.
        """
        n, d = vectors.shape

        index_class = faiss.IndexFlatIP if self.metric == 0 else faiss.IndexFlatL2
        if self.exact_index:
            index = index_class(d)
        else:
            nlist = 4 * math.ceil(n ** 0.5)
            quantiser = index_class(d)
            index = faiss.IndexIVFFlat(quantiser, d, nlist, self.metric)
            index.train(vectors)

        index.add(vectors)
        return index


# @METRICS.register_class
class PrecisionAtKMeter(IndexBasedMeter):
    def compute(self):
        scores = []
        generator = super().compute()
        for relevant_idx, closest_idx in generator:
            scores.append(precision(relevant_idx, closest_idx, k=self.k))
        return np.mean(scores)


# @METRICS.register_class
class RecallAtKMeter(IndexBasedMeter):
    def compute(self):
        scores = []
        generator = super().compute()
        for relevant_idx, closest_idx in generator:
            print(f'k = {self.k}')
            print(f'rel_indexe = {relevant_idx}')
            print(f'closest_idx = {closest_idx}')
            value = recall(relevant_idx, closest_idx, k=self.k)
            print(f'score = {value}')
            scores.append(value)
        print(f'RECALL SCORES = {scores}, mean = {np.mean(scores)}')
        return np.mean(scores)


# @METRICS.register_class
class MeanAveragePrecisionAtKMeter(IndexBasedMeter):
    def compute(self):
        scores = []
        generator = super().compute()
        for relevant_idx, closest_idx in generator:
            scores.append(average_precision(relevant_idx, closest_idx, k=self.k))
        return np.mean(scores)


# @METRICS.register_class
class NDCGAtKMeter(IndexBasedMeter):
    def compute(self):
        scores = []
        generator = super().compute()
        for relevant_idx, closest_idx in generator:
            scores.append(ndcg(relevant_idx, closest_idx, k=self.k))
        return np.mean(scores)
    