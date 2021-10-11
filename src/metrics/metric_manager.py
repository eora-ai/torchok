"""Provides class MetricManager, which stores and manages all metrics for
the model."""
from typing import List

import torch

from src.constructor.config_structure import MetricParams
from src.registry import METRICS


class MetricManager:
    phases = ['train', 'valid', 'test']
    """Manages all metrics for the model,
    stores their values at checkpoints"""

    def __init__(self, params: List[MetricParams]):
        self.metrics = {phase: {} for phase in self.phases}
        for phase in self.phases:
            for metric in params:
                if phase in metric.phases:
                    metric_obj = METRICS.get(metric.name)(**metric.params)
                    self.metrics[phase][metric_obj.name] = metric_obj

    def update(self, epoch, *args, **kwargs):
        """Update states of all metrics on training/validation loop"""
        if epoch not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of {self.phases}')
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.detach()
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.detach()

        for name, metric in self.metrics[epoch].items():
            targeted_kwargs = self.map_arguments(metric, kwargs)
            if targeted_kwargs:
                metric.update(*args, **targeted_kwargs)

    def on_epoch_end(self, epoch):
        """Summarize epoch values and return log"""
        if epoch not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of {self.phases}')
        log = {f'{epoch}/{name}': torch.tensor(metric.on_epoch_end())
               for name, metric in self.metrics[epoch].items()}
        return log

    @staticmethod
    def map_arguments(metric, kwargs):
        targeted_kwargs = {}
        for target_arg, source_arg in metric.target_fields.items():
            if source_arg in kwargs:
                arg = kwargs[source_arg]
                if not metric.use_gpu:
                    arg = arg.cpu()
                if not metric.use_torch:
                    arg = arg.numpy()
                targeted_kwargs[target_arg] = arg
        return targeted_kwargs
