from torch import Tensor, tensor, nn
from torchmetrics import Metric

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.registry import METRICS


class MetricParams:
    """
    Class for contain metric parameters.
    """
    def __init__(self, class_name: str, target_fields: dict, phases: List[str] = None, \
                 name: str = None, metric_params: dict = {}):
        """
        Args
            class_name: Metric class name with would be created.
            target_fields: Dictionary for mapping Task output with Metric forward keys.
            phases: Metric run phases.
            name: Metric name in logging output.
            metric_params: Metric class initialize parameters.
        """
        self.class_name = class_name
        self.target_fields = target_fields
        self.phases = phases if phases is not None else ['train', 'valid', 'test']
        self.name = name
        self.metric_params = metric_params


class MetricWithUtils(nn.Module):
    """
    Union class for metric and metric utils parameters
    """
    def __init__(self, metric: Metric, target_fields: Dict[str, str], name: str = None):
        """
        Args:
            metric: Metric written with TorchMetrics.
            target_fields: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
            name: The metric name.
        """
        super().__init__()
        self.__metric = metric
        self.__target_fields = target_fields
        self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def target_fields(self):
        return self.__target_fields

    def forward(self, *args, **kwargs):
        return self.__metric(*args, **kwargs)

    def compute(self):
        return self.__metric.compute()


class MetricManager(nn.Module):
    """Manages all metrics for the model."""

    # model use phases
    phases = ['train', 'valid', 'test']

    def __init__(self, params: List[MetricParams]):
        """
        Args:
            params: Metric parameters.
        """
        super().__init__()
        phase2metrics = {phase: {} for phase in self.phases}
        for phase in self.phases:
            phase2metrics[phase] = self.__get_phase_metrics(params, phase)

        self.phase2metrics = phase2metrics

    def __get_phase_metrics(self, params: List[MetricParams], phase: str) -> nn.ModuleList:
        """
        Generate metric list for current phase.

        Args:
            params: All metric params from config file.
            phase: Current phase name.

        Return:
            Metric list as nn.ModuleList for current phase. 
        """
        # create added_metric_names set
        added_metric_names = set()
        metrics = []
        for metric_params in params:
            if phase not in metric_params.phases:
                continue
            metric = METRICS.get(metric_params.class_name)(**metric_params.metric_params)
            target_fields = metric_params.target_fields
            name = metric_params.name if metric_params.name is not None else metric_params.class_name
            if name in added_metric_names:
                target_fields_values = list(target_fields.values())
                name += + '_' + target_fields_values[0] + '_' + target_fields_values[1]
            else:
                added_metric_names.add(name)

            metrics.append(MetricWithUtils(metric=metric, target_fields=target_fields, name=name))

        metrics = nn.ModuleList(metrics)
        return metrics

    def forward(self, phase: str, *args, **kwargs):
        """Update states of all metrics on phase loop.

        Args:
            phase: Phase name.
        """
        args = list(args)
        if phase not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of {self.phases}')

        for metric_with_utils in self.phase2metrics[phase]:
            targeted_kwargs = self.map_arguments(metric_with_utils.target_fields, kwargs)
            if targeted_kwargs:
                # may be we need only update because forward use compute and sync all the processses
                metric_with_utils(*args, **targeted_kwargs)
            

    def on_epoch_end(self, phase: str) -> Dict[str, Tensor]:
        """Summarize epoch values and return log.
        
        Args:
            phase: Run metric phase.

        Returns:
            Return logging dictionary, there the key is phase/metric_name and value is metric value on phase.

        Raises:
            ValueError: An error occure, when phase not in self.phases.
            ValueError: An error occure, when metric.compute() return tensor with non zero shape.
        """
        if phase not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of {self.phases}')
            
        log = {}
        for metric_with_utils in self.phase2metrics[phase]:
            metric_value = metric_with_utils.compute()
            if isinstance(metric_value, dict):
                metric_value = list(metric_value.values())[0]
            
            if len(metric_value.shape) != 0:
                raise ValueError(f'{metric_with_utils.name} must compute float value, '
                                f'not torch tensor with shap {metric_value.shape}.')
            
            log[f'{phase}/{metric_with_utils.name}'] = metric_value

        return log

    @staticmethod
    def map_arguments(metric_target_fields: Dict[str, str], task_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments between metric target_fields and task output dictionary

        Args:
            metric_target_fields: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
            task_output: Output after task forward pass.

        Return:
            Metric input dictionary like **kwargs for metric forward pass.
        """
        metric_input = {}
        for metric_target, metric_source in metric_target_fields.items():
            if metric_source in task_output:
                arg = task_output[metric_source]
                metric_input[metric_target] = arg
        return metric_input
