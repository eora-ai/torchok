import unittest

# from src.registry import METRICS
from src.metrics.metric_manager import MetricParams, MetricManager, METRICS, Phase

import torch
from torchmetrics import Metric
from typing import List, Dict

from .data_generator import FakeData, FakeDataGenerator


@METRICS.register_class
class MocSumMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('lol_sum', default=torch.tensor(0), dist_reduce_fx=None)

    def update(self, predict: torch.Tensor, target: torch.Tensor):
        self.lol_sum += 1

    def compute(self):
        return self.lol_sum


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
class MocRaiseMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, predict: torch.Tensor, target: torch.Tensor):
        return

    def compute(self, memory_block):
        return torch.tensor([1, 2])


def run_metric_manager(class_names: List[str], names: List[str], \
                       target_fields: List[Dict], data_generator: FakeDataGenerator):
    if len(class_names) != len(names) or len(names) != len(target_fields):
        raise print('Not correct Test! Input params not same length.')

    metric_params = []
    for i in range(len(class_names)):
        params = MetricParams(name=class_names[i], mapping=target_fields[i], log_name=names[i])
        metric_params.append(params)

    metric_manager = MetricManager(metric_params)
    print(f'type = {type(metric_manager)}')
    for i in range(len(data_generator)):
        metric_manager(Phase.TRAIN, **data_generator[i])

    return metric_manager.on_epoch_end(Phase.TRAIN)

class TestCase:
    def __init__(self, test_name: str, class_names: List[str], names: List[str], \
                target_fields: List[Dict], data_generator, expected):
        self.test_name = test_name
        self.class_names = class_names
        self.names = names
        self.target_fields = target_fields
        self.data_generator = data_generator
        self.expected = expected


# generate data
embedding_data = FakeData(name='embedding', shape = [512], num_repeats=5)
target_data = FakeData(name='target', shape=[10], num_repeats=5)
embedding_fake_data = [embedding_data, target_data]
data_generator = FakeDataGenerator(embedding_fake_data)
target_fields = dict(predict='embedding', target='target')


class MetricManagerTest(unittest.TestCase):
    def test(self):
        testcases = [
            TestCase(
                test_name='one_sum_metric', class_names=['MocSumMetric'], \
                names=['moc_sum'], target_fields=[target_fields], data_generator=data_generator, \
                expected={'train/moc_sum': 5}
            ),
            TestCase(
                test_name='two_metrics', \
                class_names=['MocSumMetric', 'MockConstantMetric'], \
                names=['moc_sum', None], target_fields=[target_fields, target_fields], \
                data_generator=data_generator, \
                expected={'train/moc_sum': 5, 'train/MockConstantMetric': 0}
            ),
        ]
        for case in testcases:
            print(f'case name = {case.test_name}')
            actual = run_metric_manager(
                class_names=case.class_names, names=case.names, \
                target_fields=case.target_fields, data_generator=data_generator
            )
            self.assertDictEqual(
                case.expected,
                actual,
                "failed test {} expected {}, actual {}".format(
                    case.test_name, case.expected, actual
                ),
            )


class MetricManagerRaiseTest(unittest.TestCase):
    def test(self):
        testcases = [
            TestCase(
                test_name='raise_test', \
                class_names=['MocRaiseMetric'], \
                names=[None], target_fields=[target_fields], \
                data_generator=data_generator, \
                expected={'train/MocRaiseMetric': 5}
            ),
        ]
        for case in testcases:
            with self.assertRaises(Exception):
                actual = run_metric_manager(
                    class_names=case.class_names, wrapper_names=case.wrapper_names, \
                    target_fields=case.target_fields, data_generator=data_generator
                )


if __name__ == '__main__':
    unittest.main()