import timeit
import unittest

import torch
from parameterized import parameterized

from src.registry import METRICS

num_classes = 10
metric_names = ['MeanIntersectionOverUnionMeter', 'MeanDiceMeter']
metrics = [
    ('MeanIntersectionOverUnionMeter', dict(num_classes=num_classes)),
    ('MeanIntersectionOverUnionMeter', dict(num_classes=num_classes, target_classes=[0])),
    ('MeanDiceMeter', dict(num_classes=num_classes)),
    ('MeanDiceMeter', dict(num_classes=num_classes, target_classes=[0])),
]


class TestSegmentationMetricPerformance(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.prediction = torch.rand(5, self.num_classes, 256, 256, device=self.device)
        self.target = torch.randint(0, self.num_classes, (5, 256, 256),
                                    device=self.device, dtype=torch.long)

    @parameterized.expand(metrics)
    def test_update(self, metric_name, metric_params):
        metric = METRICS.get(metric_name)(**metric_params)
        func = lambda: metric.update(target=self.target, prediction=self.prediction)
        time = timeit.timeit(func, number=1000)
        output = metric.on_epoch_end(do_reset=True)
        print(f'Execution time: {time:.4f} ms')
        print(f'Result: {output}')
        self.assertTrue(output)
        torch.cuda.empty_cache()


class TestSegmentationMetricQuality(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    @parameterized.expand(metric_names)
    def test_binary_mode(self, metric_name):
        prediction1 = torch.rand(5, 2, 256, 256, device=self.device)
        prediction2 = prediction1[:, 1] - prediction1[:, 0]
        target = torch.randint(0, 2, (5, 256, 256), device=self.device, dtype=torch.long)
        meter1 = METRICS.get(metric_name)(num_classes=2, target_classes=1)
        meter2 = METRICS.get(metric_name)(binary_mode=True)
        res1 = meter1.calculate(target=target, prediction=prediction1)
        res2 = meter2.calculate(target=target, prediction=prediction2)
        self.assertAlmostEqual(res1, res2)
        torch.cuda.empty_cache()

    @parameterized.expand(metric_names)
    def test_ignore_classes(self, metric_name):
        prediction = torch.rand(5, 2, 256, 256, device=self.device)
        target = torch.randint(0, 2, (5, 256, 256), device=self.device, dtype=torch.long)
        meter1 = METRICS.get(metric_name)(num_classes=2, target_classes=1)
        meter2 = METRICS.get(metric_name)(num_classes=2, ignore_classes=0)
        res1 = meter1.calculate(target=target, prediction=prediction)
        res2 = meter2.calculate(target=target, prediction=prediction)
        self.assertAlmostEqual(res1, res2)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main()
