from torch import Tensor, tensor, nn
from torchmetrics import Metric

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional

from src.registry import METRICS


class Phase(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


@dataclass
class MetricParams:
    """Class for contain metric parameters.

    Args:
        class_name: Metric class name with would be created.
        target_fields: Dictionary for mapping Task output with Metric forward keys.
        phases: Metric run phases.
        name: Metric name in logging output.
        metric_params: Metric class initialize parameters.
    """
    def __init__(self, class_name: str, target_fields: dict, phases: List[Phase] = None, \
                 name: str = None, metric_params: dict = {}):
        self.class_name = class_name
        self.target_fields = target_fields
        self.phases = set([Phase.TRAIN, Phase.VALID, Phase.TEST]) if phases is None else set(phases) 
        self.name = name
        self.metric_params = metric_params


class MetricWithUtils(nn.Module):
    """Union class for metric and metric utils parameters.
    
    Args:
        metric: Metric written with TorchMetrics.
        target_fields: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
        name: The metric name.
    """
    def __init__(self, metric: Metric, target_fields: Dict[str, str], name: str = None):
        super().__init__()
        self.__metric = metric
        self.__target_fields = target_fields
        self.__name = name

    @property
    def name(self) -> str:
        """The metric name in loggin."""
        return self.__name

    @property
    def target_fields(self) -> Dict[str, str]:
        """Dictionary for mapping Metric forward input keys with Task output dictionary keys."""
        return self.__target_fields

    def forward(self, *args, **kwargs):
        return self.__metric(*args, **kwargs)

    def compute(self):
        return self.__metric.compute()


class MetricManager(nn.Module):
    """Manages all metrics for the model.
    
    Args:
        params: Metric parameters.
    """
    # model use phases
    phases = [Phase.TRAIN, Phase.VALID, Phase.TEST]

    def __init__(self, params: List[MetricParams]):
        super().__init__()
        phase2metrics = {phase: {} for phase in self.phases}
        for phase in self.phases:
            phase2metrics[phase] = self.__get_phase_metrics(params, phase)

        self.__phase2metrics = phase2metrics

    def __get_phase_metrics(self, params: List[MetricParams], phase: str) -> nn.ModuleList:
        """
        Generate metric list for current phase.

        Args:
            params: All metric params from config file.
            phase: Current phase name.

        Return:
            metrics: Metric list as nn.ModuleList for current phase. 
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

    def forward(self, phase: Phase, *args, **kwargs):
        """Update states of all metrics on phase loop.

        Args:
            phase: Phase enum.
        """
        args = list(args)
        if phase not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of enum value {self.phases}')

        for metric_with_utils in self.__phase2metrics[phase]:
            targeted_kwargs = self.map_arguments(metric_with_utils.target_fields, kwargs)
            if targeted_kwargs:
                # may be we need only update because forward use compute and sync all the processses
                metric_with_utils(*args, **targeted_kwargs)
            

    def on_epoch_end(self, phase: Phase) -> Dict[str, Tensor]:
        """Summarize epoch values and return log.
        
        Args:
            phase: Run metric phase.

        Returns:
            log: Logging dictionary, there the key is phase/metric_name and value is metric value on phase.

        Raises:
            ValueError: If phase not in self.phases.
            ValueError: If metric.compute() return tensor with non zero shape.
        """
        if phase not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of enum value {self.phases}')
            
        log = {}
        for metric_with_utils in self.__phase2metrics[phase]:
            metric_value = metric_with_utils.compute()
            if isinstance(metric_value, dict):
                metric_value = list(metric_value.values())[0]
            
            if len(metric_value.shape) != 0:
                raise ValueError(f'{metric_with_utils.name} must compute float value, '
                                f'not torch tensor with shap {metric_value.shape}.')
            
            metric_key = f'{phase.value}/{metric_with_utils.name}'
            log[metric_key] = metric_value

        return log

    @staticmethod
    def map_arguments(metric_target_fields: Dict[str, str], task_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments between metric target_fields and task output dictionary

        Args:
            metric_target_fields: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
            task_output: Output after task forward pass.

        Returns:
            metric_input: Metric input dictionary like **kwargs for metric forward pass.
        """
        metric_input = {}
        for metric_target, metric_source in metric_target_fields.items():
            if metric_source in task_output:
                arg = task_output[metric_source]
                metric_input[metric_target] = arg
        return metric_input

    @property
    def phase2metrics(self) -> Dict[Phase, nn.ModuleList]:
        """Dictionary of phase to their metrics list with type nn.ModuleList([MetricWithUtils])"""
        return self.__phase2metrics
