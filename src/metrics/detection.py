from src.registry import METRICS
from .common import Metric
import torch
from torchmetrics import MAP

@METRICS.register_class
class MeanAveragePrecision(Metric):
    def __init__(self, name, target_fields = None, metric_name='map_50'):
        #metric names from https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/detection/map.py
        super().__init__(name=name, target_fields=target_fields)
        self.metric_name = metric_name
        self.map = MAP()
        self.use_gpu = True
        self.use_torch = True

    def calculate(self, target, prediction):
        # Update metric with predictions and respective ground truth
        self.map.update(prediction, target)
        result = self.map.compute()
        if result is None:
            return 0
        return result[self.metric_name]

    def update(self, target, prediction, *args, **kwargs):
        """Updates metric buffer"""
        batch_size = prediction['boxes'].shape[0]
        value = self.calculate([target], [prediction]) * batch_size
        self.mean = (self.n * self.mean + float(value)) / (self.n + batch_size)
        self.n += batch_size
