import torch
from torch import Tensor, tensor
from torchmetrics import Metric

from typing import List, Dict, Optional


class MetricMemoryBank(Metric):
    #!TODO add documentation
    def __init__(self, tracked_obj_names: List[str], **kwargs):
        super().__init__(**kwargs)
        self.tracked_obj_names = tracked_obj_names
        for name in tracked_obj_names:
            self.add_state(name, default=[], dist_reduce_fx=None)
        
    def update(self, **output):
        for name, value in output.items():
            if name in self.tracked_obj_names:
                if isinstance(value, torch.Tensor):
                    value = value.cpu()
                self.__getattribute__(name).append(value)

    def compute(self):
        obj_name2memory = {}
        for name in self.tracked_obj_names:
            obj_name2memory[name] = torch.cat(self.__getattribute__(name), 0)
        return obj_name2memory


class MetricWrapper:
    #!TODO add documentation
    def __init__(self, metric: Metric, target_fields: Dict, name: str = None, phases: List[str] = ['train', 'valid']):
        self.metric = metric
        self.name = name if name is not None else type(metric).__name__
        self.phases = phases
        self.require_memory_bank = True if hasattr(metric, 'require_memory_bank') else False
        self.target_fields = target_fields

    def update(self, *args, **kwargs):
        self.metric.update(args, kwargs)

    def compute(self, memory_bank: Optional[MetricMemoryBank] = None):
        if self.require_memory_bank:
            metric_value = self.metric.compute(memory_bank)
        else:
            metric_value = self.metric.compute()
        
        extension_name = ''
        if isinstance(metric_value, dict):
            if self.name in metric_value:
                extension_name = '_' + self.name
                metric_value = metric_value[self.name]
            else:
                extension_name = '_' + list(metric_value.keys())[0]
                metric_value = list(metric_value.values())[0]

        if not isinstance(metric_value, torch.Tensor):
            metric_value = torch.tensor(metric_value)
            
        return extension_name, metric_value
    