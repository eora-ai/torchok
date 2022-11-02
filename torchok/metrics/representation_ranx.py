import numpy as np
from ranx.metrics import average_precision, hit_rate, ndcg, precision, recall
from sklearn.preprocessing import normalize
from typing import Callable, Generator, List, Optional, Tuple, Union

from torchok.metrics.index_base_metric import DatasetType, IndexBasedMeter
from torchok.constructor import METRICS


__all__ = [
    'PrecisionAtKMeter',
    'RecallAtKMeter',
    'MeanAveragePrecisionAtKMeter',
    'NDCGAtKMeter',
]


class RanxBasedMeter(IndexBasedMeter):
    def __init__(self, exact_index: bool, dataset_type: str, metric_distance: str,
                 metric_func: Callable, k: Optional[int] = None, search_batch_size: Optional[int] = None,
                 normalize_vectors: bool = False, target_averaging: bool = False, k_as_target_len: bool = False,
                 **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=metric_func, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)

    def process_data_for_metric_func(self, closest_scores: np.ndarray, closest_idxs: np.ndarray,
                                     relevants_idxs: np.ndarray, query_col_idxs: np.ndarray,
                                     scores: np.ndarray, k: int) -> List:
        # NDCG score=distance is needed to sort more relevant examples, but in this part of code we had
        # already sorted our examples by faiss. So if we change score = 1 to distance with type float
        # the index of relevant will be also float and after that inside ranx it may be fail to compare
        # relevant int index with our relevant float index.
        closest_idxs = map(lambda idx:
                            np.stack((closest_idxs[idx], [1] * len(closest_idxs[idx])), axis=1),
                            np.arange(len(closest_idxs)))

        if query_col_idxs is None:
            search_relevants_idxs = map(lambda r: np.stack((r, np.ones_like(r)), axis=1), relevants_idxs)
        else:
            # clear_query_order_numbers cleared of queries, so it has local indexes and the elements can be
            # obtained like clear_query_order_numbers[batch_idxs]
            search_relevants_idxs = map(lambda r_q:
                                        np.stack((r_q[0], scores[r_q[0], r_q[1]]), axis=1),
                                        zip(relevants_idxs, query_col_idxs))

        search_relevants_idxs = list(search_relevants_idxs)
        closest_idxs = list(closest_idxs)
        return list(search_relevants_idxs, closest_idxs, k-1)


@METRICS.register_class
class HitAtKMeter(RanxBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=hit_rate, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class PrecisionAtKMeter(RanxBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=precision, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RecallAtKMeter(RanxBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=recall, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class MeanAveragePrecisionAtKMeter(RanxBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=average_precision, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class NDCGAtKMeter(RanxBasedMeter):
    def __init__(self, dataset_type: str, exact_index: bool = True,
                 metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=ndcg, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)
