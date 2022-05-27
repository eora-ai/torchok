import unittest
import numpy as np
import os
from typing import *

from src.metrics.representation import RecallAtKMeter, PrecisionAtKMeter, MeanAveragePrecisionAtKMeter, NDCGAtKMeter
from src.metrics.representation import DatasetType, MetricDistance

from .context import *


CPU_COUNT = os.cpu_count()


class TestDDPRepresentationMetrics(unittest.TestCase):
    def test_ddp_mode_when_metric_recall_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_precision_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = PrecisionAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = ANSWERS['precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_average_precision_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = MeanAveragePrecisionAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = ANSWERS['average_precision']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_ddp_mode_when_metric_ndcg_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = NDCGAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = ANSWERS['ndcg']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])

    def test_search_batch_size_when_ddp_mode_metric_recall_data_representation(self):
        if CPU_COUNT < 3:
            return
        metric_class = RecallAtKMeter
        metric_params = {
            'dataset_type': DatasetType.REPRESENTATION,
            'search_batch_size': 2
        }
        trainer_params = dict(accelerator="cpu", strategy="ddp", num_processes=3)
        metrics = run_model(metric_class, metric_params, trainer_params)
        answer = ANSWERS['recall']
        for k in range(1, MAX_K + 1):
            np.testing.assert_almost_equal(answer[k], metrics[k])
    