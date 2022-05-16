from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


# Phase utils
class Phase(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    PREDICT = 'predict'

phase_mapping = {
    'train': Phase.TRAIN,
    'valid': Phase.VALID,
    'test': Phase.TEST,
    'predict': Phase.PREDICT
}


# Optimization parameters
@dataclass
class OptmizerParams:
    name: str
    params: Dict = field(default_factory=dict)
    
@dataclass
class SchedulerPLParams:
    interval: Optional[str] = None
    monitor: Optional[str] = None

@dataclass
class SchedulerParams:
    name: str
    pl_params: Optional[SchedulerPLParams] = None
    params: Dict = field(default_factory=dict)

@dataclass
class OptimizationParams:
    optimizer: OptmizerParams
    scheduler: Optional[SchedulerParams]


# Data parameters
@dataclass
class AugmentationParams:
    name: str
    params: Dict = field(default_factory=dict)

@dataclass
class DatasetParams:
    name: str
    params: Dict = field(default_factory=dict)
    transform: Optional[List[AugmentationParams]] = field(default_factory=list)
    augment: Optional[List[AugmentationParams]] = field(default_factory=list)

@dataclass
class DataParams:
    dataset: DatasetParams
    dataloader: Dict = field(default_factory=dict)

@dataclass
class DataloaderParams:
    # I think it must be list, with Enum Phase inside DataParams
    train: Optional[List[DataParams]]
    valid: Optional[List[DataParams]]
    test: Optional[List[DataParams]]
    predict: Optional[List[DataParams]]


# Losses parameters
@dataclass
class LossParams:
    name: str
    tag: str
    mapping: Dict[str, str]
    params: Dict = field(default_factory=dict)
    weight: Optional[float] = None

@dataclass
class JointLossParams:
    loss_params: List[LossParams]
    normalize_weights: bool = True


# Metric parameters
@dataclass
class MetricParams:
    name: str
    mapping: Dict[str, str]
    params: Dict = field(default_factory=dict)
    phases: Optional[List[Any]] = field(default_factory=list)
    prefix: Optional[str] = None


# Config parameters
@dataclass
class ConfigParams:
    data: DataloaderParams
    optimization: List[OptimizationParams]
    losses: JointLossParams
    metrics: List[MetricParams] = field(default_factory=list)

    def __post_init__(self):
        """Post process for metrics phases. 
        
        Convert string phase to enum.
        Hydra can't call __post_init__ of it's fields, beacuse it's actually OmegaConf dict or list.
        And hydra can't handle list of Enums. It's force to converts values to Enums.

        Raises:
            KeyError: If phase in config not in mapping dict.
        """
        for i in range(len(self.metrics)):
            current_phases = self.metrics[i].phases
            if len(current_phases) == 0:
                self.metrics[i].phases = [Phase.TRAIN, Phase.VALID, Phase.TEST, Phase.PREDICT]
            else:
                new_phases = []
                for phase in current_phases:
                    if phase not in phase_mapping:
                        raise KeyError(f'Phase has no key = {phase}, it must be one of {list(phase_mapping.keys())}')
                    else:
                        new_phases.append(phase_mapping[phase])
                self.metrics[i].phases = new_phases
