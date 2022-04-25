import unittest

import torch
from torchmetrics import Metric
from typing import List, Dict, Union
import numpy as np
from src.metrics.representation import RecallAtKMeter, PrecisionAtKMeter, MeanAveragePrecisionAtKMeter, NDCGAtKMeter


vectors = {
    'classification': torch.tensor([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], \
        [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 0]]),

    'representation': torch.tensor([
        [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], # queries
        [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], # database
    ])
}

targets = {
    'classification': torch.tensor([0, 1, 1, 2, 2, 1, 0, 0, 3]),
    'representation': torch.tensor([0, 1, 2, 3, 1, 2, 1, 0, 0])
}

is_queries = torch.tensor([True, True, True, True, False, False, False, False, False])

scores = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 2.5, 0, 0],
        [0, 0, 3, 0],
        [0, 1, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0]
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
    def __init__(self, test_name, dataset: str = 'classification', search_batch_size: bool = None, \
                 exact_index: bool = True, normalize_input: bool = True, metric: str = 'IP'):
        self.params = dict(
            dataset = dataset,
            search_batch_size = search_batch_size,
            exact_index = exact_index,
            normalize_input = normalize_input,
            metric = metric,
        )
        self.class_name = test_name
        if dataset == 'classification':
            self.expected = classification_dataset_answers[test_name]
        else:
            self.expected = representation_dataset_answers[test_name]


def compute_metric_dict(test_case: TestCase):
    metric_class = name2class[test_case.class_name]
    answer_dict = {}
    for k in range(1, 5):
        metric = metric_class(**test_case.params, k=k)
        for i in range(3):
            vec = vectors[test_case.params['dataset']][3*i : 3*(i + 1)]
            target = targets[test_case.params['dataset']][3*i : 3*(i + 1)]

            score = None
            is_query = None
            if test_case.params['dataset'] == 'representation':
                score = scores[3*i : 3*(i + 1)]
                is_query = is_queries[3*i : 3*(i + 1)]

            metric.update(vectors=vec, targets=target, scores=score, is_queries=is_query)
        value = metric.compute()
        answer_dict[k] = float(value)
    # print(f'ANSWER DICT = {answer_dict}')
    return answer_dict


class TestRepresentationMetrics(unittest.TestCase):
    def test_classification_dataset(self):
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

    def test_representation_dataset(self):
        test_cases = [
            TestCase(test_name='recall', dataset='representation'),
            TestCase(test_name='precision', dataset='representation'),
            TestCase(test_name='average_precision', dataset='representation'),
            TestCase(test_name='ndcg', dataset='representation'),
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

    def test_classification_dataset_different_search_size(self):
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

    def test_representation_dataset_different_search_size(self):
        test_cases = [
            TestCase(test_name='recall', dataset='representation', search_batch_size=1),
            TestCase(test_name='recall', dataset='representation', search_batch_size=2),
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
