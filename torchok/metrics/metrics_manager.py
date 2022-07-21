import numbers
import numpy as np

from torch import Tensor, tensor, nn
from torchmetrics import Metric

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

from torchok.constructor import METRICS
from torchok.constructor.config_structure import MetricParams, Phase


class MetricWithUtils(nn.Module):
    """Union class for metric and metric utils parameters."""
    def __init__(self, metric: Metric, mapping: Dict[str, str], log_name: str):
        """Initalize MetricWithUtils.
        
        Args:
            metric: Metric written with TorchMetrics.
            mapping: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
            log_name: The metric name used in logs.
        """
        super().__init__()
        self.__metric = metric
        self.__mapping = mapping
        self.__log_name = log_name

    @property
    def log_name(self) -> str:
        """The metric name used in logs."""
        return self.__log_name

    @property
    def mapping(self) -> Dict[str, str]:
        """Dictionary for mapping Metric forward input keys with Task output dictionary keys."""
        return self.__mapping

    def forward(self, *args, **kwargs):
        return self.__metric(*args, **kwargs)

    def compute(self):
        return self.__metric.compute()


class MetricsManager(nn.Module):
    """Manages all metrics for a Task."""
    def __init__(self, params: List[MetricParams]):
        """Initialize MetricManager.

        Args:
            params: Metric parameters.
        """
        super().__init__()
        self.__phase2metrics = nn.ModuleDict()
        for phase in Phase:
            self.__phase2metrics[phase.name] = self.__get_phase_metrics(params, phase) 

    def __get_phase_metrics(self, params: List[MetricParams], phase: Phase) -> nn.ModuleList:
        """Generate metric list for current phase.

        Args:
            params: All metric params from config file.
            phase: Current phase Enum.

        Returns:
            metrics: Metric list as nn.ModuleList for current phase.

        Raises:
            ValueError: If got two identical log_names.
        """
        # create added_metric_names list
        added_log_names = []
        metrics = []
        for metric_params in params:
            if phase not in metric_params.phases:
                continue
            metric = METRICS.get(metric_params.name)(**metric_params.params)
            mapping = metric_params.mapping
            prefix = '' if metric_params.prefix is None else metric_params.prefix + '_'
            log_name = prefix + metric_params.name
            if log_name in added_log_names:
                raise ValueError(f'Got two metrics with identical names: {log_name}. '
                                 f'Please, set differet prefixes for identical metrics in the config file.')           
            else:
                added_log_names.append(log_name)

            metrics.append(MetricWithUtils(metric=metric, mapping=mapping, log_name=log_name))

        metrics = nn.ModuleList(metrics)

        return metrics

    def forward(self, phase: Phase, *args, **kwargs):
        """Update states of all metrics on phase loop.

        Args:
            phase: Phase Enum.
        """
        args = list(args)

        for metric_with_utils in self.__phase2metrics[phase.name]:
            targeted_kwargs = self.map_arguments(metric_with_utils.mapping, kwargs)
            metric_with_utils(*args, **targeted_kwargs)
            
    def on_epoch_end(self, phase: Phase) -> Dict[str, Tensor]:
        """Summarize epoch values and return log.
        
        Args:
            phase: Run metric phase.

        Returns:
            log: Logging dictionary, where key is `<phase>/<metric_name>` and value is metric value for
            a given phase.

        Raises:
            ValueError: If metric.compute() returns not numerical value.
        """ 
        log = {}
        for metric_with_utils in self.__phase2metrics[phase.name]:
            metric_value = metric_with_utils.compute()
            # If it tensor type with wrong shape.
            if isinstance(metric_value, Tensor) and len(metric_value.shape) != 0:
                raise ValueError(f'{metric_with_utils.log_name} must compute number value, ' 
                                 f'not torch tensor with shape {metric_value.shape}.')
            # If it numpy array with wrong shape.
            if isinstance(metric_value, np.ndarray) and len(metric_value.shape) != 0:
                raise ValueError(f'{metric_with_utils.log_name} must compute number value, ' 
                                 f'not numpy array with shape {metric_value.shape}.')
            # If it numpy array with one element but wrong dtype
            if (isinstance(metric_value, np.ndarray) and len(metric_value.shape) == 0 and 
                np.issubdtype(metric_value.dtype, np.number)):
                raise ValueError(f'{metric_with_utils.log_name} must compute number value, ' 
                                 f'not numpy array element with dtype {metric_value.dtype}.')

            is_number = isinstance(metric_value, numbers.Number)
            # If not numeric type.
            if not (is_number or isinstance(metric_value, Tensor) or  isinstance(metric_value, np.ndarray)):
                raise ValueError(f'{metric_with_utils.log_name} must compute number value, ' 
                                 f'not numpy array element with dtype {metric_value.dtype}.')
            
            metric_key = f'{phase.value}/{metric_with_utils.log_name}'
            log[metric_key] = metric_value

        return log

    @staticmethod
    def map_arguments(mapping: Dict[str, str], task_output: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: create a common function for MetricManager and Constructor
        """Map arguments between metric target_fields and task output dictionary.

        Args:
            mapping: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
            task_output: Output after task forward pass.

        Returns:
            metric_input: Metric input dictionary like ``**kwargs`` for metric forward pass.

        Raises:
            ValueError: If not found mapping_source in task_output keys.
        """
        metric_input = {}
        for metric_target, metric_source in mapping.items():
            if metric_source in task_output:
                arg = task_output[metric_source]
                metric_input[metric_target] = arg
            else:
                raise ValueError(f'Cannot find {metric_source} for your mapping {metric_target} : {metric_source}. '
                                 f'You should either add {metric_source} output to your model or remove the mapping '
                                 f'from configuration')
        return metric_input

    @property
    def phase2metrics(self) -> Dict[Phase, nn.ModuleList]:
        """Dictionary of phase to their metrics list with type nn.ModuleList([MetricWithUtils])"""
        return self.__phase2metrics
