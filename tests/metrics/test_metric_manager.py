import unittest
# !TODO WHY its now import without it
import sys
sys.path.append('../../')

import torch
from torchmetrics import Metric
from typing import List, Dict

from src.registry import METRICS
from src.metrics.metric_manager import MetricManager
from data_generator import FakeData, FakeDataGenerator


class MetricParams:
    def __init__(self, class_name: str, wrapper_params: dict = {}, metric_params: dict = {}):
        self.class_name = class_name
        self.wrapper_params = wrapper_params
        self.metric_params = metric_params

    
def generate_wrapper_params(target_fields, name=None, phases=['train', 'valid']):
    wrapper_dict = {
        'target_fields': target_fields,
        'name': name,
        'phases': phases,
    }
    return wrapper_dict


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
class MocRequoreMemoryBlockMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.require_memory_bank = True

    def update(self, predict: torch.Tensor, target: torch.Tensor):
        return

    def compute(self, memory_block):
        shape_dict = {}
        for name, value in memory_block.items():
            shape_dict[name] = value.shape[0]
        return shape_dict


@METRICS.register_class
class MocRaiseMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, predict: torch.Tensor, target: torch.Tensor):
        return

    def compute(self, memory_block):
        return torch.tensor([1, 2])


def run_metric_manager(class_names: List[str], wrapper_names: List[str], \
                       target_fields: List[Dict], data_generator: FakeDataGenerator):
    if len(class_names) != len(wrapper_names) or len(wrapper_names) != len(target_fields):
        raise print('Not correct Test! Input params not same length.')

    metric_params = []
    for i in range(len(class_names)):
        wrapper_params = generate_wrapper_params(target_fields[i], wrapper_names[i])
        params = MetricParams(class_name=class_names[i], wrapper_params=wrapper_params)
        metric_params.append(params)

    metric_manager = MetricManager(metric_params)

    for i in range(len(data_generator)):
        metric_manager.update('train', **data_generator[i])

    return metric_manager.on_epoch_end('train')

class TestCase:
    def __init__(self, test_name: str, class_names: List[str], wrapper_names: List[str], \
                target_fields: List[Dict], data_generator, expected):
        self.test_name = test_name
        self.class_names = class_names
        self.wrapper_names = wrapper_names
        self.target_fields = target_fields
        self.data_generator = data_generator
        self.expected = expected

    
class MetricManagerTest(unittest.TestCase):
    def test(self):
        # generate data
        embedding_data = FakeData(name='embedding', shape = [512])
        target_data = FakeData(name='target', shape=[10])
        embedding_fake_data = [embedding_data, target_data]
        sizes = [5, 5]
        data_generator = FakeDataGenerator(embedding_fake_data, sizes)

        

        target_fields = target_fields = dict(predict='embedding', target='target')
        testcases = [
            TestCase(
                test_name='one_sum_metric', class_names=['MocSumMetric'], \
                wrapper_names=['moc_sum'], target_fields=[target_fields], data_generator=data_generator, \
                expected={'train/moc_sum': 5}
            ),
            TestCase(
                test_name='one_mem_block_metric', class_names=['MocRequoreMemoryBlockMetric'], \
                wrapper_names=['moc_mem_block'], target_fields=[target_fields], data_generator=data_generator, \
                expected={'train/moc_mem_block_embedding': 20}
            ),
            TestCase(
                test_name='two_metrics', \
                class_names=['MocSumMetric', 'MocRequoreMemoryBlockMetric'], \
                wrapper_names=['moc_sum', 'moc_mem_block'], target_fields=[target_fields, target_fields], \
                data_generator=data_generator, \
                expected={'train/moc_sum': 5, 'train/moc_mem_block_embedding': 20}
            ),
            TestCase(
                test_name='check_names_with_two_metrics', \
                class_names=['MocSumMetric', 'MocRequoreMemoryBlockMetric'], \
                wrapper_names=[None, None], target_fields=[target_fields, target_fields], \
                data_generator=data_generator, \
                expected={'train/MocSumMetric': 5, 'train/MocRequoreMemoryBlockMetric_embedding': 20}
            ),
        ]
        for case in testcases:
            actual = run_metric_manager(
                class_names=case.class_names, wrapper_names=case.wrapper_names, \
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
        # generate data
        embedding_data = FakeData(name='embedding', shape = [512])
        target_data = FakeData(name='target', shape=[10])
        embedding_fake_data = [embedding_data, target_data]
        sizes = [5, 5]
        data_generator = FakeDataGenerator(embedding_fake_data, sizes)

        target_fields = target_fields = dict(predict='embedding', target='target')
        testcases = [
            TestCase(
                test_name='raise_test', \
                class_names=['MocRaiseMetric'], \
                wrapper_names=[None], target_fields=[target_fields], \
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