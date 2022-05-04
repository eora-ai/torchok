from torch import Tensor, tensor, nn
from torchmetrics import Metric

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

from src.registry import METRICS


# Metric parameters
class Phase(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    PREDICT = 'predict'

phase_mapping = {
    'train': Phase.TRAIN,
    'valid': Phase.VALID,
    'test': Phase.TEST,
    'predict': Phase.PREDICT,
    'Phase.TRAIN': Phase.TRAIN,
    'Phase.VALID': Phase.VALID,
    'Phase.TEST': Phase.TEST,
    'Phase.PREDICT': Phase.PREDICT,
}

@dataclass
class MetricParams:
    name: str
    mapping: Dict[str, str]
    log_name: str = None
    params: Dict = field(default_factory=dict)
    phases: List[Phase] = None

    def __post_init__(self):
        """Post process for phases. 
        
        Hydra can't handle list of Enums. It's force to converts values to Enums.

        Raises:
            KeyError: If phase in config not in mapping dict.
        """
        if self.phases is None:
            self.phases = [Phase.TRAIN, Phase.VALID, Phase.TEST, Phase.PREDICT]
        else:
            phases = []
            for phase in self.phases:
                if phase not in phase_mapping:
                    raise KeyError(f'Phase has no key = {phase}, it must be one of {list(phase_mapping.keys())}')
                else:
                    phases.append(phase_mapping[phase])
            self.phases = phases


class MetricWithUtils(nn.Module):
    """Union class for metric and metric utils parameters.
    
    Args:
        metric: Metric written with TorchMetrics.
        mapping: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
        log_name: The metric name used in logs.
    """
    def __init__(self, metric: Metric, mapping: Dict[str, str], log_name: str):
        super().__init__()
        self.__metric = metric
        self.__mapping = mapping
        self.__log_name = log_name

    @property
    def log_name(self) -> str:
        """The metric name used in loggs."""
        return self.__log_name

    @property
    def mapping(self) -> Dict[str, str]:
        """Dictionary for mapping Metric forward input keys with Task output dictionary keys."""
        return self.__mapping

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
    phases = [Phase.TRAIN, Phase.VALID, Phase.TEST, Phase.PREDICT]

    def __init__(self, params: List[MetricParams]):
        super().__init__()
        # Change list to set in phases
        for param in params:
            param.phases = set(param.phases)

        phase2metrics = {phase: {} for phase in self.phases}
        for phase in self.phases:
            phase2metrics[phase] = self.__get_phase_metrics(params, phase)

        self.__phase2metrics = phase2metrics

    def __get_phase_metrics(self, params: List[MetricParams], phase: Phase) -> nn.ModuleList:
        """
        Generate metric list for current phase.

        Args:
            params: All metric params from config file.
            phase: Current phase Enum.

        Returns:
            metrics: Metric list as nn.ModuleList for current phase. 
        """
        # create added_metric_names set
        added_log_names = set()
        metrics = []
        for metric_params in params:
            if phase not in metric_params.phases:
                continue
            metric = METRICS.get(metric_params.name)(**metric_params.params)
            mapping = metric_params.mapping
            log_name = metric_params.log_name if metric_params.log_name is not None else metric_params.name
            if log_name in added_log_names:
                mapping_values = list(mapping.values())
                log_name += + '_' + mapping_values[0] + '_' + mapping_values[1]
            else:
                added_log_names.add(log_name)

            metrics.append(MetricWithUtils(metric=metric, mapping=mapping, log_name=log_name))

        metrics = nn.ModuleList(metrics)
        return metrics

    def forward(self, phase: Phase, *args, **kwargs):
        """Update states of all metrics on phase loop.

        Args:
            phase: Phase Enum.
        """
        args = list(args)
        if phase not in self.phases:
            raise ValueError(f'Incorrect epoch setting. '
                             f'Please choose one of enum value {self.phases}')

        for metric_with_utils in self.__phase2metrics[phase]:
            targeted_kwargs = self.map_arguments(metric_with_utils.mapping, kwargs)
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
                raise ValueError(f'{metric_with_utils.log_name} must compute float value, '
                                f'not torch tensor with shap {metric_value.shape}.')
            
            metric_key = f'{phase.value}/{metric_with_utils.log_name}'
            log[metric_key] = metric_value

        return log

    @staticmethod
    def map_arguments(mapping: Dict[str, str], task_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map arguments between metric target_fields and task output dictionary

        Args:
            mapping: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
            task_output: Output after task forward pass.

        Returns:
            metric_input: Metric input dictionary like **kwargs for metric forward pass.
        """
        metric_input = {}
        for metric_target, metric_source in mapping.items():
            if metric_source in task_output:
                arg = task_output[metric_source]
                metric_input[metric_target] = arg
        return metric_input

    @property
    def phase2metrics(self) -> Dict[Phase, nn.ModuleList]:
        """Dictionary of phase to their metrics list with type nn.ModuleList([MetricWithUtils])"""
        return self.__phase2metrics
