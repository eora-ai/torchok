import torch
import numpy as np

from torchmetrics import (RetrievalFallOut, RetrievalHitRate, RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG,
                          RetrievalPrecision, RetrievalPrecisionRecallCurve, RetrievalRPrecision, RetrievalRecall)
from typing import List, Optional, Any, Dict

from torchok.metrics.index_base_metric import IndexBasedMeter
from torchok.constructor import METRICS


__all__ = [
    'RetrievalFallOutMeter',
    'RetrievalHitRateMeter',
    'RetrievalMAPMeter',
    'RetrievalMRRMeter',
    'RetrievalNormalizedDCGMeter',
    'RetrievalPrecisionMeter',
    'RetrievalPrecisionRecallCurveMeter',
    'RetrievalRPrecisionMeter',
    'RetrievalRecallMeter'
]


class TorchMetricBaseMetr(IndexBasedMeter):
    def __init__(self, exact_index: bool, dataset_type: str, metric_distance: str,
                 metric_class: type, metric_params: Optional[Dict[str, Any]] = None,
                 k: Optional[int] = None, search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False,
                 **kwargs):
        metric_params = metric_params if metric_params is not None else dict()
        metric_func = metric_class(**metric_params)
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_func=metric_func, k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)

    def process_data_for_metric_func(self, closest_scores: np.ndarray, closest_idxs: np.ndarray,
                                     relevants_idxs: np.ndarray, query_col_idxs: np.ndarray,
                                     scores: np.ndarray, k: int) -> List:
        preds = torch.tensor(closest_scores)
        target = torch.tensor([np.isin(closest_idxs[i], relevants_idxs[i]) for i in range(len(closest_idxs))],
                              dtype=torch.long)
        indexes = torch.tensor([target.shape[1] * [i] for i in range(len(target))], dtype=torch.long)
        return [preds, target, indexes]


@METRICS.register_class
class RetrievalFallOutMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalFallOut, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RetrievalHitRateMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalHitRate, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RetrievalMAPMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalMAP, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RetrievalMRRMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalMRR, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RetrievalNormalizedDCGMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalNormalizedDCG, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RetrievalPrecisionMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalPrecision, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RetrievalPrecisionRecallCurveMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalPrecisionRecallCurve, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RetrievalRPrecisionMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalRPrecision, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)


@METRICS.register_class
class RetrievalRecallMeter(TorchMetricBaseMetr):
    def __init__(self, dataset_type: str, metric_params: Optional[Dict[str, Any]] = None,
                 exact_index: bool = True, metric_distance: str = 'IP', k: Optional[int] = None,
                 search_batch_size: Optional[int] = None, normalize_vectors: bool = False,
                 target_averaging: bool = False, k_as_target_len: bool = False, **kwargs):
        super().__init__(exact_index=exact_index, dataset_type=dataset_type, metric_distance=metric_distance,
                         metric_class=RetrievalRecall, metric_params=metric_params,
                         k=k, search_batch_size=search_batch_size,
                         normalize_vectors=normalize_vectors, target_averaging=target_averaging,
                         k_as_target_len=k_as_target_len, **kwargs)