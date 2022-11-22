import math
from abc import ABC
from enum import Enum
from typing import Callable, Generator, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from torchmetrics import Metric


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
    Compute method return generator with relevant and closest (FAISS searched) indexes. The relevant index contain
    its relevant index with scores for current query. And the closest contain the closest index with its distance.
    """
    full_state_update: bool = False

    def __init__(self, exact_index: bool, dataset_type: str, metric_distance: str,
                 metric_func: Callable, k_as_target_len: bool = False, k: Optional[int] = None,
                 use_batching_search: bool = True, search_batch_size: Optional[int] = None,
                 normalize_vectors: bool = False, group_averaging: bool = False, raise_empty_query: bool = True,
                 **kwargs):
        """Initialize IndexBasedMeter.

        Args:
            exact_index: If true then build fair inner product or Euclidean index (depends on the metric chosen),
                otherwise the index will be an approximate nearest neighbors search index.
            dataset_type: Dataset type (classification or representation), which will be used to calculate metric.
            metric_distance: Metric distance (IP - cosine distance, L2 - euclidean), which will be used to build
                FAISS index.
            metric_func: Representation metric (e.g. ranx metric function) with the follow backend
                `def metric_func(
                    qrels: Union[np.ndarray, numba.typed.List],
                    run: Union[np.ndarray, numba.typed.List],
                    k: int = 0,
                ) -> np.ndarray`
                where qrels - y_true and run - y_pred
                see https://github.com/AmenRa/ranx/blob/ccab1549de81e7366e34213c86089e965db55f72/ranx/metrics.py
                for more details.
            k_as_target_len: If true will be search with different top k, where these different k is the length of each
                uniq target vectors. If true parameter k will be not use.
            k: Number of top closest indexes to get. If k_as_target_len is true this value will not be used.
            use_batching_search: If true will do one request at a time for every query and if `group_averaging`
                parameter is true will do one request at a time for each group. Otherwise, will use batch for each
                query request.
            search_batch_size: The size for one FAISS search request, default = num CPUs.
            normalize_vectors: If true vectors will be normalized, otherwise no.
            group_averaging: If true compute metric averaging by the targets.
            raise_empty_query: If true raise in case when dataset has query without relevants.

        Raises:
            ValueError: If metric or dataset is not correct write.
        """
        super().__init__(**kwargs)
        self.exact_index = exact_index

        self.dataset_type = dataset_enum_mapping[dataset_type]
        self.metric_distance = distance_enum_mapping[metric_distance]
        self.metric_func = metric_func
        self.normalize_vectors = normalize_vectors
        self.group_averaging = group_averaging
        self.k_as_target_len = k_as_target_len
        self.use_batching_search = use_batching_search
        self.raise_empty_query = raise_empty_query
        # set search_batch_size as num CPUs if search_batch_size is None
        self.search_batch_size = torch.get_num_threads() if search_batch_size is None else search_batch_size

        # set k as 1 if k is None
        k = 1 if k is None else k
        # as queries vectors can be relevant to other query vectors, so we should search k + 1 and delete first
        # and remove the first found vector because it would be a query
        self.search_k = k + 1
        # but in metric compute study, k must not change
        self.metric_compute_k = k

        self.add_state('vectors', default=[], dist_reduce_fx=None)
        if self.dataset_type == DatasetType.CLASSIFICATION:
            # if classification dataset
            self.add_state('group_labels', default=[], dist_reduce_fx=None)
        else:
            # if representation dataset
            self.add_state('query_idxs', default=[], dist_reduce_fx=None)
            self.add_state('scores', default=[], dist_reduce_fx=None)
            self.add_state('group_labels', default=[], dist_reduce_fx=None)

    def update(self, vectors: torch.Tensor, group_labels: Optional[torch.Tensor] = None,
               query_idxs: Optional[torch.Tensor] = None, scores: Optional[torch.Tensor] = None):
        """Append tensors in storage.

        Args:
            vectors: Often it would be embeddings, size (batch_size, embedding_size).
            group_labels: The labels for every vector in classification mode, size (batch_size).
            query_idxs: Integer tensor where values >= 0 represent indices of queries with corresponding
                vectors in vectors tensor and value -1 indicates that the corresponding vector isn't a query.
            scores: The score tensor, see representation dataset for more information,
                size (batch_size, total_num_queries).

        Raises:
            ValueError: If dataset is of classification type and targets is None, or if dataset is of representation
                type and at least one of scores or query_idxs is None.
        """
        vectors = vectors.detach().cpu()
        self.vectors.append(vectors)
        if self.dataset_type == DatasetType.CLASSIFICATION:
            if group_labels is None:
                raise ValueError("In classification dataset group_labels must be not None.")
            group_labels = group_labels.detach().cpu()
            self.group_labels.append(group_labels)
        else:
            if query_idxs is None:
                raise ValueError("In representation dataset query_numbers must be not None.")
            if scores is None:
                raise ValueError("In representation dataset scores must be not None")

            query_idxs = query_idxs.detach().cpu()
            self.query_idxs.append(query_idxs)

            scores = scores.detach().cpu()
            self.scores.append(scores)

            group_labels = group_labels.detach().cpu()
            self.group_labels.append(group_labels)

    def compute(self) -> float:
        """Compute metric value.

        Firstly it gathers all tensors in storage (done by torchmetrics).
        Then it prepares data, separates query and database vectors.
        Then it builds the FAISS index.
        Then, it creates a generator of relevant and closest arrays.
        Finally, it compute metric.

        Returns:
            metric: Metric value.
        """
        vectors = torch.cat(self.vectors).numpy()
        if self.normalize_vectors:
            vectors = normalize(vectors)

        if self.dataset_type == DatasetType.CLASSIFICATION:
            # if classification dataset
            group_labels = torch.cat(self.group_labels).numpy()
            # prepare data
            relevant_idxs, faiss_vector_idxs, \
                query_row_idxs, query_as_relevant = self.prepare_classification_data(group_labels)
            # mock scores and query column indexes because it belongs to representation data
            scores = None
            query_column_idxs = None
        else:
            # if representation dataset
            scores = torch.cat(self.scores).numpy()
            query_idxs = torch.cat(self.query_idxs).numpy()
            group_labels = torch.cat(self.group_labels).numpy()
            # prepare data
            relevant_idxs, faiss_vector_idxs, query_column_idxs, \
                query_row_idxs, query_as_relevant = self.prepare_representation_data(query_idxs, scores)

        # build index
        vectors = vectors.astype(np.float32)
        index = self.build_index(vectors[faiss_vector_idxs])

        # split query by group_label if metric compute target averaging
        if self.group_averaging:
            uniq_group_labels = np.unique(group_labels)
            group_indexes_split = np.array([np.where(group_labels == label)[0] for label in uniq_group_labels])
        else:
            group_indexes_split = np.arange(len(group_labels))[None]

        # compute metric
        metric = []
        for group_indexes in group_indexes_split:
            curr_target_metric = 0
            query_target_idxs = np.isin(query_row_idxs, group_indexes)
            curr_query_col_idxs = None if query_column_idxs is None else query_column_idxs[query_target_idxs]
            curr_relevant_idxs = relevant_idxs[query_target_idxs]
            curr_query_row_idxs = query_row_idxs[query_target_idxs]
            curr_query_as_relevant = query_as_relevant[query_target_idxs]

            if self.k_as_target_len:
                # + 1 because query can be in index
                # and - count of queries which are not in index
                k = len(group_indexes) + 1 - len(np.where(~curr_query_as_relevant)[0])
            else:
                k = self.search_k

            # create relevant, closest generator
            generator = self.query_generator(index, vectors, curr_relevant_idxs,
                                             curr_query_row_idxs, faiss_vector_idxs,
                                             curr_query_as_relevant, k,
                                             scores, curr_query_col_idxs)
            for batch_size, args in generator:
                curr_target_metric += batch_size * self.metric_func(*args).mean()

            curr_target_metric /= len(curr_query_row_idxs)
            metric.append(curr_target_metric)

        metric = float(np.mean(metric))
        return metric

    def process_data_for_metric_func(self, closest_scores: np.ndarray, closest_idxs: np.ndarray,
                                     relevants_idxs: np.ndarray, query_col_idxs: np.ndarray,
                                     scores: np.ndarray, k: int) -> List:
        """Process obtained data after faiss search for metric function. Output of this function will be use like
        *args for self.metric_func and will be call in self.compute() method

        Args:
            closest_scores: Faiss found the closest scores.
            closest_idxs: Faiss found the closest reference indexes.
            relevants_idxs: Relevant indexes.
            query_col_idxs: Queries column indexes in scores matrix.
            scores: Scores matrix.
            k: Number of top closest indexes to get.
        """
        pass

    def prepare_representation_data(self, query_idxs: np.ndarray, scores: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in representation dataset case.

        Separate query and database vectors from storage vectors.
        Prepare scores.
        Generate relevant indexes for every query request.

        Args:
            query_idxs: Integer array where values >= 0 represent indices of queries with corresponding
                vectors in vectors tensor and value -1 indicates that the corresponding vector isn't a query. Also
                value of this array is the column number in the score matrix, which need to reproduce
                relevant elements.
            scores: The score tensor, see representation dataset for more information,
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
            faiss_vector_idxs: Array of indexes in vectors storage which would be in faiss index.
            query_column_idxs: Array of queries column indexes in scores matrix. It means query order number in origin
                dataset, because after shuffle dataset the query order may be change we need to store it to reproduce
                for every query vector it's relevant.
            query_row_idxs: Array of queries row indexes in scores matrix, lso this index belong to vector storage,
                need to get query vector in search study.
            query_as_relevant: Array of query row indexes which are in relevant, i.e. belong to queries and
                gallery simultaneously.

        Raises:
            ValueError: If dataset has query vector without relevants.
        """
        is_query = query_idxs >= 0
        # array of columns indexes in scores matrix, need to get relevant
        query_column_idxs = query_idxs[is_query]
        # array of row indexes in scores matrix, also this index belong to vector storage, need to get query vector in
        # search study
        query_row_idxs = np.where(is_query)[0]

        # found query row indexes which are in relevant, i.e. belong to queries and relevants simultaneously
        query_as_relevant = np.any(scores[query_row_idxs, :] > 0, axis=-1)

        faiss_vector_idxs = np.arange(len(scores))
        # remove query indexes which is not relevant for another query from faiss_vector_idxs
        clear_query_idxs = query_row_idxs[~query_as_relevant]
        faiss_vector_idxs = np.delete(faiss_vector_idxs, clear_query_idxs)

        relevant_idxs = []
        for query_col_idx in query_column_idxs:
            curr_relevant_idxs = np.where(scores[:, query_col_idx] > 0.)[0]
            if len(curr_relevant_idxs) == 0:
                if self.raise_empty_query:
                    raise ValueError('Representation metric. The dataset contains a query vector that does not '
                                     'has relevants. Set parameter raise_empty_query to False for compute.')
                relevant_idxs.append([])
            else:
                # Need to sort relevant indexes by its scores for NDCG metric
                current_scores = scores[curr_relevant_idxs, query_col_idx]
                sort_indexes = np.argsort(current_scores)
                curr_relevant_idxs = curr_relevant_idxs[sort_indexes[::-1]]
                relevant_idxs.append(curr_relevant_idxs)

        relevant_idxs = np.array(relevant_idxs)
        return relevant_idxs, faiss_vector_idxs, query_column_idxs, query_row_idxs, query_as_relevant

    def prepare_classification_data(self, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for faiss build index, and following search, in classification dataset case.

        In the classification case, ll the vectors would be used as queries, and the relevant vectors will be vectors
        that have the same label as the query.

        Args:
            targets: Targets in classification task for every vector.

        Returns:
            relevant_idxs: Array of relevant indexes in gallery data, for every query.
            gallery_idxs: Array of gallery indexes in vectors storage.
            query_row_idxs: Array of queries indexes in vectors storage.
            query_as_relevant: Array of query row indexes which are in relevant, i.e. belong to queries and
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
                if len(relevant) == 0 and self.raise_empty_query:
                    raise ValueError(f'Representation metric. The class {groups.index[i]} has only one element.')
                query_row_idxs.append(query_idx)
                relevant_idxs.append(relevant)

        relevant_idxs = np.array(relevant_idxs, dtype=object)
        query_row_idxs = np.array(query_row_idxs)
        gallery_idxs = np.arange(len(targets))
        # all query vectors is in relevant, so create array with True elements
        query_as_relevant = np.full((len(gallery_idxs),), fill_value=True, dtype=np.bool)
        return relevant_idxs, gallery_idxs, query_row_idxs, query_as_relevant

    def clear_faiss_output(self, faiss_output: np.ndarray, query_as_relevant: np.ndarray) -> np.ndarray:
        """Remove first element from faiss output array if query in index, and remove last element because
        this class search k + 1 element.

        Args:
            faiss_output: Faiss output array which must be handled.
            query_as_relevant: Boolean array of indexes which indicates if query is in relevant set,
                i.e. belong to queries and gallery simultaneously.

        Returns:
            faiss_output: Handled faiss output.
        """
        # need delete first elements where query in faiss index i.e. batch_query_as_relevant==True
        faiss_output_delete_firs = np.delete(faiss_output[query_as_relevant], 0, axis=1)

        # and delete last element where query is not in faiss index i.e. batch_query_as_relevant==False,
        # because we use k + 1 for search
        faiss_output_delete_last = np.delete(faiss_output[~query_as_relevant], -1, axis=1)

        q_count, k = faiss_output.shape
        faiss_output = np.zeros((q_count, k - 1))

        faiss_output[query_as_relevant] = faiss_output_delete_firs
        faiss_output[~query_as_relevant] = faiss_output_delete_last
        return faiss_output

    def query_generator(self, index: Union[faiss.IndexFlatIP, faiss.IndexFlatL2],
                        vectors: np.ndarray, relevants_idxs: np.ndarray,
                        query_row_idxs: np.ndarray, faiss_vector_idxs: np.ndarray, query_as_relevant: np.ndarray,
                        k: int, scores: Optional[np.ndarray] = None, query_col_idxs: Optional[np.ndarray] = None
                        ) -> Generator[Tuple[int, List], None, None]:
        """Create inputs *args for metric function, by faiss index search.

        This function use self.search_batch_size to define how many vectors to send per one faiss search request in
        case when self.use_batch_searching = True.
        If self.use_batch_searching is False will do one faiss search request.

        Need to know, that query_row_idxs, query_col_idxs and query_as_relevant - have the same size, and for i-th
        query element the following information is available:
            query_row_idxs[i] - index to get embedding vector in vectors storage
            query_col_idxs[i] - column index in scores matrix, need to get score value for every relevant (used in NDCG)
            query_as_relevant[i] - whether the query in relevant i.e. in gallery data, need to remove first element in
                retrieved indexes if it true, and last element if it false (because in fact, k + 1 search request is
                being made)

        Args:
            index: Faiss database built index.
            vectors: All embedding vectors, it contains gallery and queries vectors.
            relevants_idxs: Array of relevant indexes for every query, index belong vectors storage.
            query_row_idxs: Array of query indexes in vectors storage.
            faiss_vector_idxs: Array of indexes in vectors storage which is in faiss index.
            query_as_relevant: Boolean array of indexes which indicates if query is in relevant set,
                i.e. belong to queries and gallery simultaneously.
            k: Number of top closest indexes to get.
            scores: Array of scores.
            query_col_idxs: Array of query row indexes which are in relevant, i.e. belong to queries and
                gallery simultaneously.

        Returns:
            Generator which contain current batch value and self.process_data_for_metric_func output.

            batch value - current batch value.
            metric_input_list - self.process_data_for_metric_func output.
        """
        search_batch_size = self.search_batch_size if self.use_batching_search else len(query_row_idxs)
        for i in range(0, len(query_row_idxs), search_batch_size):
            batch_idxs = np.arange(i, min(i + search_batch_size, len(query_row_idxs)))

            batch_query_as_relevant = query_as_relevant[batch_idxs]
            batch_query_row_idxs = query_row_idxs[batch_idxs]
            batch_relevants_idxs = relevants_idxs[batch_idxs]
            if scores is None or query_col_idxs is None:
                batch_query_col_idxs = None
            else:
                batch_query_col_idxs = query_col_idxs[batch_idxs]

            # queries_idxs - indexes of vectors (global indexes), so queries vectors can be obtained
            # like vectors[queries_idxs[batch_idxs]]
            batch_closest_scores, local_closest_idxs = index.search(vectors[batch_query_row_idxs], k=k)

            # get global indexes
            batch_closest_idxs = faiss_vector_idxs[local_closest_idxs]

            # remove first element from faiss output if query in faiss index and remove last element
            # because k + 1 element searched
            batch_closest_scores = self.clear_faiss_output(batch_closest_scores, batch_query_as_relevant)
            batch_closest_idxs = self.clear_faiss_output(batch_closest_idxs, batch_query_as_relevant)

            metric_input_list = self.process_data_for_metric_func(closest_scores=batch_closest_scores,
                                                                  closest_idxs=batch_closest_idxs,
                                                                  relevants_idxs=batch_relevants_idxs,
                                                                  query_col_idxs=batch_query_col_idxs,
                                                                  scores=scores, k=k)
            yield len(batch_idxs), list(metric_input_list)

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
            quantizer = index_class(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, self.metric_distance.value)
            index.train(vectors)

        index.add(vectors)
        return index
