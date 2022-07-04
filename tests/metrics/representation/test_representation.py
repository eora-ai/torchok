import unittest
import numpy as np
import os

from src.metrics.representation import RecallAtKMeter, PrecisionAtKMeter, MeanAveragePrecisionAtKMeter, NDCGAtKMeter

from .data import CLASSIFICATION_ANSWERS, REPRESENTATION_ANSWERS
from .context import run_model, MAX_K

CPU_COUNT = os.cpu_count()


class TestRepresentationMetrics(unittest.TestCase):
    def test_precision_when_dataset_is_representation(self):
        metric_class = PrecisionAtKMeter
        metric_params = {
            'dataset_type': 'representation',
        }
        metrics = run_model(metric_class, metric_params)
        answer = REPRESENTATION_ANSWERS['precision']
        
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
    
    def test_precision_when_dataset_is_classification(self):
        metric_class = PrecisionAtKMeter
        metric_params = {
            'dataset_type': 'classification',
            'normalize_vectors': True
        }
        metrics = run_model(metric_class, metric_params)
        answer = CLASSIFICATION_ANSWERS['precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
    
    def test_recall_when_dataset_is_representation(self):
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': 'representation',
        }
        metrics = run_model(metric_class, metric_params)
        answer = REPRESENTATION_ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
    
    def test_recall_when_dataset_is_classification(self):
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': 'classification',
            'normalize_vectors': True
        }
        metrics = run_model(metric_class, metric_params)
        answer = CLASSIFICATION_ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_average_precision_when_dataset_is_representation(self):
        metric_class = MeanAveragePrecisionAtKMeter
        metric_params = {
            'dataset_type': 'representation',
        }
        metrics = run_model(metric_class, metric_params)
        answer = REPRESENTATION_ANSWERS['average_precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_average_precision_when_dataset_is_classification(self):
        metric_class = MeanAveragePrecisionAtKMeter
        metric_params = {
            'dataset_type': 'classification',
            'normalize_vectors': True
        }
        metrics = run_model(metric_class, metric_params)
        answer = CLASSIFICATION_ANSWERS['average_precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ndcg_when_dataset_is_representation(self):
        metric_class = NDCGAtKMeter
        metric_params = {
            'dataset_type': 'representation',
        }
        metrics = run_model(metric_class, metric_params)
        answer = REPRESENTATION_ANSWERS['ndcg']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
