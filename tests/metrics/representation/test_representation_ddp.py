import unittest
import numpy as np
import os

from torchok.metrics.representation import RecallAtKMeter, PrecisionAtKMeter, MeanAveragePrecisionAtKMeter, NDCGAtKMeter

from .data import CLASSIFICATION_ANSWERS, REPRESENTATION_ANSWERS, REPRESENTATION_QUERY_AS_RELEVANT_ANSWERS
from .context import run_model, MAX_K


CPU_COUNT = os.cpu_count()


class TestDDPRepresentationMetrics(unittest.TestCase):
    def test_ddp_mode_when_metric_recall_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': 'representation',
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = REPRESENTATION_ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_recall_data_representation_with_query_as_relevant(self):
        if CPU_COUNT < 3:
            return
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': 'representation',
            'normalize_vectors': True,
            'score_type': 'query_as_relevant'
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = REPRESENTATION_QUERY_AS_RELEVANT_ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_recall_data_classification(self):
        if CPU_COUNT < 3:
            return
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': 'classification',
            'normalize_vectors': True
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = CLASSIFICATION_ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_precision_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = PrecisionAtKMeter
        metric_params = {
            'dataset_type': 'representation',
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = REPRESENTATION_ANSWERS['precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_precision_data_representation_with_query_as_relevant(self):
        if CPU_COUNT < 3:
            return
        metric_class = PrecisionAtKMeter
        metric_params = {
            'dataset_type': 'representation',
            'normalize_vectors': True,
            'score_type': 'query_as_relevant'
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = REPRESENTATION_QUERY_AS_RELEVANT_ANSWERS['precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_precision_data_classification(self):
        if CPU_COUNT < 3:
            return
        metric_class = PrecisionAtKMeter
        metric_params = {
            'dataset_type': 'classification',
            'normalize_vectors': True
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = CLASSIFICATION_ANSWERS['precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_average_precision_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = MeanAveragePrecisionAtKMeter
        metric_params = {
            'dataset_type': 'representation',
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = REPRESENTATION_ANSWERS['average_precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_average_precision_data_classification(self):
        if CPU_COUNT < 3:
            return
        metric_class = MeanAveragePrecisionAtKMeter
        metric_params = {
            'dataset_type': 'classification',
            'normalize_vectors': True
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = CLASSIFICATION_ANSWERS['average_precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_ndcg_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = NDCGAtKMeter
        metric_params = {
            'dataset_type': 'representation',
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = REPRESENTATION_ANSWERS['ndcg']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_search_batch_size_when_ddp_mode_metric_recall_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': 'representation',
            'search_batch_size': 2
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = REPRESENTATION_ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
