import torch
from torch import Tensor, tensor
from torchmetrics import Metric

from typing import List, Dict, Optional

from src.registry import METRICS
from .metric_utils import MetricMemoryBank, MetricWrapper

class MetricManager:
    phases = ['train', 'valid', 'test']
    """Manages all metrics for the model,
    stores their values at checkpoints"""

    def __init__(self, params: List):
        # !TODO append List[MetricParams]
        self.metrics = {phase: {} for phase in self.phases}
        self.memory_bank = {phase: None for phase in self.phases}
        for phase in self.phases:
            phase_mem_bank_name_list = []
            for metric in params:
                if phase in metric.wrapper_params['phases']:
                    metric_obj = METRICS.get(metric.class_name)(**metric.metric_params)
                    metric_wrapper = MetricWrapper(metric=metric_obj, **metric.wrapper_params)
                    self.metrics[phase][metric_wrapper.name] = metric_wrapper
                    if metric_wrapper.require_memory_bank:
                        for target_arg, source_arg in metric_wrapper.target_fields.items():
                            if source_arg not in phase_mem_bank_name_list:
                                phase_mem_bank_name_list.append(source_arg)
            # create memory bank
            if len(phase_mem_bank_name_list) != 0:
                self.memory_bank[phase] = MetricMemoryBank(phase_mem_bank_name_list)

    def update(self, phase, *args, **kwargs):
        """Update states of all metrics on training/validation loop"""
        if phase not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of {self.phases}')
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = arg.detach()
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.detach()
        
        if self.memory_bank[phase] is not None:
            self.memory_bank[phase].update(**kwargs)
        
        for name, metric in self.metrics[phase].items():
            targeted_kwargs = self.map_arguments(metric, kwargs)
            if targeted_kwargs: 
                metric.update(*args, **targeted_kwargs)

    def on_epoch_end(self, phase):
        """Summarize epoch values and return log"""
        if phase not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of {self.phases}')
        phase_memory_bank = None
        if self.memory_bank[phase] is not None:
            phase_memory_bank = self.memory_bank[phase].compute()
            
        log = {}
        for name, metric in self.metrics[phase].items():
            if hasattr(metric, 'require_memory_bank'):
                extension_name, metric_value = metric.compute(phase_memory_bank)
            else:
                extension_name, metric_value = metric.compute()
            
            metric_value_shape = list(metric_value.shape)
            
            if len(metric_value_shape) != 0:
                raise Exception(
                    f'{metric.name} must compute float value, not torch tensor with shap {metric_value_shape}'
                    )
            
            log[f'{phase}/{name}{extension_name}'] = metric_value

        return log

    @staticmethod
    def map_arguments(metric, kwargs):
        targeted_kwargs = {}
        for target_arg, source_arg in metric.target_fields.items():
            if source_arg in kwargs:
                arg = kwargs[source_arg]
                targeted_kwargs[target_arg] = arg
        return targeted_kwargs
