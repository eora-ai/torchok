import numbers
from typing import Any, Dict, List

import numpy as np
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric

from torchok.constructor import METRICS
from torchok.constructor.config_structure import MetricParams, Phase


class MetricWithUtils(nn.Module):
    """Union class for metric and metric utils parameters."""

    def __init__(self, metric: Metric, mapping: Dict[str, str], log_name: str, dataloader_idx: int):
        """Initialize MetricWithUtils.

        Args:
            metric: Metric written with TorchMetrics.
            mapping: Dictionary for mapping Metric forward input keys with Task output dictionary keys.
            log_name: The metric name used in logs.
            dataloader_idx: Dataloader index on which metrics are calculated.
        """
        super().__init__()
        self.metric = metric
        self.mapping = mapping
        self.log_name = log_name
        self.dataloader_idx = dataloader_idx

    def map_arguments(self, task_output: Dict[str, Any]) -> Dict[str, Any]:
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
        for metric_target, metric_source in self.mapping.items():
            if metric_source in task_output:
                arg = task_output[metric_source]
                metric_input[metric_target] = arg
            else:
                raise ValueError(f'Cannot find {metric_source} for your mapping {metric_target} : {metric_source}. '
                                 f'You should either add {metric_source} output to your model or remove the mapping '
                                 f'from configuration')
        return metric_input

    def update(self, dataloader_idx: int = 0, **kwargs):
        """Update metric states if current `dataloader_idx` equal `self.dataloader_idx`.

        Add `*args` and `**kwargs` (usually it is batch) to current state.

        Args:
            dataloader_idx: Current dataloader index.
        """
        if dataloader_idx == self.dataloader_idx:
            targeted_kwargs = self.map_arguments(kwargs)
            self.metric.update(**targeted_kwargs)

    def compute(self):
        """Compute metric on the whole current state."""
        value = self.metric.compute()
        return value

    def reset(self):
        """Reset metric states."""
        self.metric.reset()


class MetricsManager(nn.Module):
    """Manages all metrics for a Task."""

    def __init__(self, params: List[MetricParams]):
        """Initialize MetricManager.

        Args:
            params: Metric parameters.
        """
        super().__init__()
        self.phase2metrics = nn.ModuleDict()
        for phase in Phase:
            self.phase2metrics[phase.name] = self._get_phase_metrics(params, phase)

    def _get_phase_metrics(self, params: List[MetricParams], phase: Phase) -> nn.ModuleList:
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
            mapping = metric_params.mapping

            # create base log name, it would be use as log name if metric compute for one dataloder
            base_log_name = metric_params.name if metric_params.tag is None else metric_params.tag

            # Metric manager support many dataloaders only for Validation and Test Phases
            if phase == Phase.VALID:
                dataloader_idxs = metric_params.val_dataloader_idxs
            elif phase == Phase.TEST:
                dataloader_idxs = metric_params.test_dataloader_idxs
            else:
                dataloader_idxs = [0]

            if phase in [Phase.VALID, Phase.TEST] and len(dataloader_idxs) > 1:
                # but if metric compute for many dataloders -> log name = '{base_log_name}_{dataloader_idx}'
                log_names = [f'{base_log_name}_dataloader_{dataloader_idx}' for dataloader_idx in dataloader_idxs]
            else:
                log_names = [base_log_name]

            for log_name in log_names:
                if log_name in added_log_names:
                    raise ValueError(f'Got two metrics with identical names: {log_name}. '
                                     f'Please, set different prefixes for identical metrics in the config file.')
                else:
                    added_log_names.append(log_name)

            # add metric for each dataloader index
            for dataloader_idx, log_name in zip(dataloader_idxs, log_names):
                metric = METRICS.get(metric_params.name)(**metric_params.params)
                metrics.append(MetricWithUtils(metric=metric, mapping=mapping,
                                               log_name=log_name, dataloader_idx=dataloader_idx))

        metrics = nn.ModuleList(metrics)

        return metrics

    def update(self, phase: Phase, dataloader_idx: int = 0, **kwargs):
        """Update states of all metrics on phase loop.

        MetricsManager update method use only update method of metrics. Because metric forward method
        increases computation time (see MetricWithUtils forward method for more information).

        Args:
            phase: Phase Enum.
        """
        for metric_with_utils in self.phase2metrics[phase.name]:
            metric_with_utils.update(dataloader_idx, **kwargs)

    @staticmethod
    def is_number(num: Any) -> bool:
        if isinstance(num, np.ndarray):
            return len(num.shape) == 0 and np.issubdtype(num.dtype, np.number)
        elif isinstance(num, Tensor):
            return len(num.shape) == 0
        else:
            return isinstance(num, numbers.Number)

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
        for metric_with_utils in self.phase2metrics[phase.name]:
            metric_value = metric_with_utils.compute()
            if isinstance(metric_value, dict):
                metric_keys = list(metric_value.keys())
                for metric_name_d in metric_keys:
                    metric_value_d = metric_value.pop(metric_name_d)
                    if self.is_number(metric_value_d):
                        metric_value[f'{phase.value}/{metric_with_utils.log_name}_{metric_name_d}'] = metric_value_d
                # If there is no numeric value
                if len(metric_value) == 0:
                    raise ValueError(f'Metric manager on_epoch_end method. Metric {metric_with_utils.log_name}'
                                     f'return dict with has no numeric values.')
                log.update(metric_value)
            elif self.is_number(metric_value):
                metric_key = f'{phase.value}/{metric_with_utils.log_name}'
                log[metric_key] = metric_value
            else:
                raise ValueError(f'Metric manager on_epoch_end method. Metric {metric_with_utils.log_name} '
                                 f'return no numeric value.')

            # Do reset
            metric_with_utils.reset()

        return log
