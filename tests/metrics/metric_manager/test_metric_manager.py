import unittest

from src.constructor import METRICS
from src.metrics.metrics_manager import MetricParams, MetricManager, Phase

import torch
from torchmetrics import Metric
from typing import List, Dict

from .data_generator import FakeData, FakeDataGenerator


@METRICS.register_class
class MockSumMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('sum', default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, predict: torch.Tensor, target: torch.Tensor):
        self.sum += 1

    def compute(self):
        return self.sum


@METRICS.register_class
class MockConstantMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('constant', default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, predict: torch.Tensor, target: torch.Tensor):
        return

    def compute(self):
        return self.constant


@METRICS.register_class
class MockRaiseMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, predict: torch.Tensor, target: torch.Tensor):
        return

    def compute(self, memory_block):
        return torch.tensor([1, 2])


def run_metric_manager(names: List[str], prefixes: List[str], \
                       mappings: List[Dict], data_generator: FakeDataGenerator):
    if len(names) != len(prefixes) or len(prefixes) != len(mappings):
        raise ValueError('Not correct Test! Input params not same length.')

    metric_params = []
    for i in range(len(names)):
        params = MetricParams(name=names[i], mapping=mappings[i], prefix=prefixes[i])
        metric_params.append(params)

    metric_manager = MetricManager(metric_params)
    print(f'type = {type(metric_manager)}')
    for i in range(len(data_generator)):
        metric_manager(Phase.TRAIN, **data_generator[i])

    return metric_manager.on_epoch_end(Phase.TRAIN)


class TestCase:
    def __init__(self, test_name: str, names: List[str], prefixes: List[str], \
                mappings: List[Dict], data_generator, expected):
        self.test_name = test_name
        self.names = names
        self.prefixes = prefixes
        self.mappings = mappings
        self.data_generator = data_generator
        self.expected = expected


# generate data
embedding_data = FakeData(name='embedding', shape = [512], num_repeats=5)
target_data = FakeData(name='target', shape=[10], num_repeats=5)
embedding_fake_data = [embedding_data, target_data]
data_generator = FakeDataGenerator(embedding_fake_data)
mappings = dict(predict='embedding', target='target')


class MetricManagerTest(unittest.TestCase):
    def test_metrics_manager_when_one_metric_is_defined_sum_metric(self):
        case = TestCase(
            test_name='one_sum_metric', names=['MockSumMetric'], \
            prefixes=['moc_sum'], mappings=[mappings], data_generator=data_generator, \
            expected={'train/moc_sum_MockSumMetric': 5}
            )
    
        print(f'case name = {case.test_name}')
        actual = run_metric_manager(
            names=case.names, prefixes=case.prefixes, \
            mappings=case.mappings, data_generator=data_generator
        )
        self.assertDictEqual(
            case.expected,
            actual,
            "failed test {} expected {}, actual {}".format(
                case.test_name, case.expected, actual
            ),
        )
    def test_metrics_manager_when_two_metrics_is_defined_sum_metric_and_constant_metric(self):
        case = TestCase(
            test_name='two_metrics', \
            names=['MockSumMetric', 'MockConstantMetric'], \
            prefixes=['moc_sum', None], mappings=[mappings, mappings], \
            data_generator=data_generator, \
            expected={'train/moc_sum_MockSumMetric': 5, 'train/MockConstantMetric': 0}
            )
        
        print(f'case name = {case.test_name}')
        actual = run_metric_manager(
            names=case.names, prefixes=case.prefixes, \
            mappings=case.mappings, data_generator=data_generator
        )
        self.assertDictEqual(
            case.expected,
            actual,
            "failed test {} expected {}, actual {}".format(
                case.test_name, case.expected, actual
            ),
        )

class MetricManagerRaiseTest(unittest.TestCase):
    def test_metrics_manager_when_raise_is_happend(self):
        testcases = [
            TestCase(
                test_name='raise_test', \
                names=['MockRaiseMetric'], \
                prefixes=[None], mappings=[mappings], \
                data_generator=data_generator, \
                expected={'train/MocRaiseMetric': 5}
            ),
        ]
        for case in testcases:
            with self.assertRaises(Exception):
                actual = run_metric_manager(
                    names=case.names, prefixes=case.prefixes, \
                    mappings=case.mappings, data_generator=data_generator
                )


if __name__ == '__main__':
    unittest.main()
    