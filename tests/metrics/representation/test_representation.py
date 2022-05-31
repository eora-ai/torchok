import unittest
import numpy as np
import os
from typing import *

from src.metrics.representation import RecallAtKMeter, PrecisionAtKMeter, MeanAveragePrecisionAtKMeter, NDCGAtKMeter
from src.metrics.representation import DatasetType, MetricDistance

from .context import *


CPU_COUNT = os.cpu_count()


class TestRepresentationMetrics(unittest.TestCase):
    def test_precision_when_dataset_is_representation(self):
        metric_class = PrecisionAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
        }
        metrics = run_model(metric_class, metric_params)
        answer = ANSWERS['precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
    
    def test_precision_when_dataset_is_classification(self):
        metric_class = PrecisionAtKMeter
        metric_params = {
            'dataset_type': DatasetType.CLASSIFICATION,
        }
        metrics = run_model(metric_class, metric_params)
        answer = ANSWERS['precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
    
    def test_recall_when_dataset_is_representation(self):
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
        }
        metrics = run_model(metric_class, metric_params)
        answer = ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
    
    def test_recall_when_dataset_is_classification(self):
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': DatasetType.CLASSIFICATION,
        }
        metrics = run_model(metric_class, metric_params)
        answer = ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_average_precision_when_dataset_is_representation(self):
        metric_class = MeanAveragePrecisionAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
        }
        metrics = run_model(metric_class, metric_params)
        answer = ANSWERS['average_precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_average_precision_when_dataset_is_classification(self):
        metric_class = MeanAveragePrecisionAtKMeter
        metric_params = {
            'dataset_type': DatasetType.CLASSIFICATION,
        }
        metrics = run_model(metric_class, metric_params)
        answer = ANSWERS['average_precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ndcg_when_dataset_is_representation(self):
        metric_class = NDCGAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
        }
        metrics = run_model(metric_class, metric_params)
        answer = ANSWERS['ndcg']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_search_bach_size_when_metric_recall_and_data_representation(self):
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
            'search_batch_size': 2
        }
        metrics = run_model(metric_class, metric_params)
        answer = ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
    