import unittest

import torch
from torchmetrics import Metric
from typing import List, Dict, Union
import numpy as np
from src.metrics.representation import RecallAtKMeter, PrecisionAtKMeter, MeanAveragePrecisionAtKMeter, NDCGAtKMeter
from src.metrics.representation import DatasetType, MetricDistance


vectors = {
    DatasetType.CLASSIFICATION: torch.tensor([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], \
        [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]]),

    DatasetType.REPRESENTATION: torch.tensor([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], \
        [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]]),
}

targets = torch.tensor([0, 1, 1, 2, 2, 1, 0, 0, 3], dtype=torch.int32)

queries_idxs = torch.tensor([0, 1, -1, 2, -1, -1, -1, -1, 3], dtype=torch.int32)

scores = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 2.5, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 1, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [0, 0, 0, 0],
    ]
)

classification_dataset_answers = {
    'recall': {
        1: 0.5, 
        2: 2/3, 
        3: 2.5/3,
        4: 1 
    },
    'precision': {
        1: 2/3,
        2: 0.5,
        3: 4/9,
        4: 5/12
    },
    'average_precision': {
        1: 0.5,
        2: 0.5,
        3: 0.7222222222222223,
        4: 0.8055555555555555
    },
    'ndcg': {
        1: 0.6666666666666666,
        2: 0.5436432511904858,
        3: 0.7688578654609097,
        4: 0.8568805729851068
    }
}

representation_dataset_answers = {
    'recall': {
        1: 0.5, 
        2: 2/3, 
        3: 2.5/3,
        4: 1 
    },
    'precision': {
        1: 2/3,
        2: 0.5,
        3: 4/9,
        4: 5/12
    },
    'average_precision': {
        1: 0.5,
        2: 0.5,
        3: 0.7222222222222223,
        4: 0.8055555555555555
    },
    'ndcg': {
        1: 0.6666666666666666,
        2: 0.4847038505208375,
        3: 0.6933556704671239,
        4: 0.8099261657231794
    }
}

name2class = {
    'recall': RecallAtKMeter,
    'precision': PrecisionAtKMeter,
    'average_precision': MeanAveragePrecisionAtKMeter,
    'ndcg': NDCGAtKMeter
}


class TestCase:
    def __init__(self, test_name, dataset_type: DatasetType = DatasetType.CLASSIFICATION, \
            search_batch_size: bool = None, exact_index: bool = True, normalize_vectors: bool = True, \
            metric_distance: MetricDistance = MetricDistance.IP):
        self.params = dict(
            dataset_type = dataset_type,
            search_batch_size = search_batch_size,
            exact_index = exact_index,
            normalize_vectors = normalize_vectors,
            metric_distance = metric_distance,
        )
        self.class_name = test_name
        if dataset_type == DatasetType.CLASSIFICATION:
            self.expected = classification_dataset_answers[test_name]
        else:
            self.expected = representation_dataset_answers[test_name]


def compute_metric_dict(test_case: TestCase):
    metric_class = name2class[test_case.class_name]
    answer_dict = {}
    for k in range(1, 5):
        metric = metric_class(**test_case.params, k=k)
        for i in range(3):
            vec = vectors[test_case.params['dataset_type']][3*i : 3*(i + 1)]
            target = targets[3*i : 3*(i + 1)]

            curr_scores = None
            curr_queries_idxs = None
            if test_case.params['dataset_type'] == DatasetType.REPRESENTATION:
                curr_scores = scores[3*i : 3*(i + 1)]
                curr_queries_idxs = queries_idxs[3*i : 3*(i + 1)]

            metric.update(vectors=vec, targets=target, scores=curr_scores, queries_idxs=curr_queries_idxs)
        value = metric.compute()
        answer_dict[k] = float(value)
    return answer_dict


class TestRepresentationMetrics(unittest.TestCase):
    def test_all_metrics_when_classification_dataset_is_used(self):
        test_cases = [
            TestCase(test_name='recall'),
            TestCase(test_name='precision'),
            TestCase(test_name='average_precision'),
            TestCase(test_name='ndcg'),
        ]

        for case in test_cases:
            actual = compute_metric_dict(case)

            self.assertDictEqual(
                case.expected,
                actual,
                "failed test {} expected {}, actual {}".format(
                    case.class_name, case.expected, actual
                ),
            )

    def test_all_metrics_when_representation_dataset_is_used(self):
        test_cases = [
            TestCase(test_name='recall', dataset_type=DatasetType.REPRESENTATION),
            TestCase(test_name='precision', dataset_type=DatasetType.REPRESENTATION),
            TestCase(test_name='average_precision', dataset_type=DatasetType.REPRESENTATION),
            TestCase(test_name='ndcg', dataset_type=DatasetType.REPRESENTATION),
        ]

        for case in test_cases:
            actual = compute_metric_dict(case)

            self.assertDictEqual(
                case.expected,
                actual,
                "failed test {} expected {}, actual {}".format(
                    case.class_name, case.expected, actual
                ),
            )

    def test_recall_when_different_search_batch_size_and_classification_dataset_was_define(self):
        test_cases = [
            TestCase(test_name='recall', search_batch_size=1),
            TestCase(test_name='recall', search_batch_size=2),
        ]

        for case in test_cases:
            actual = compute_metric_dict(case)

            self.assertDictEqual(
                case.expected,
                actual,
                "failed test {} expected {}, actual {}".format(
                    case.class_name, case.expected, actual
                ),
            )

    def test_recall_when_different_search_batch_size_and_representation_dataset_was_define(self):
        test_cases = [
            TestCase(test_name='recall', dataset_type=DatasetType.REPRESENTATION, search_batch_size=1),
            TestCase(test_name='recall', dataset_type=DatasetType.REPRESENTATION, search_batch_size=2),
        ]

        for case in test_cases:
            actual = compute_metric_dict(case)

            self.assertDictEqual(
                case.expected,
                actual,
                "failed test {} expected {}, actual {}".format(
                    case.class_name, case.expected, actual
                ),
            )

if __name__ == '__main__':
    unittest.main()
