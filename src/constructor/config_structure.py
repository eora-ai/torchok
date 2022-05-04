from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class Phase(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    PREDICT = 'predict'


# Optimization parameters
@dataclass
class OptmizerParams:
    name: str
    params: Dict = field(default_factory=dict)
    
@dataclass
class SchedulerPLParams:
    interval: int
    monitor: str # I think need ENUM

@dataclass
class SchedulerParams:
    name: str
    pl_params: SchedulerPLParams
    params: Dict = field(default_factory=dict)

@dataclass
class OptimizationParams:
    optimizer: OptmizerParams
    scheduler: SchedulerParams


# Data parameters
@dataclass
class DatasetParams:
    name: str
    params: dict

@dataclass
class AugmentationParams:
    name: str
    params: Dict = field(default_factory=dict)

@dataclass
class DataParams:
    dataset: DatasetParams
    transforms: List[AugmentationParams]
    augment: Optional[List[AugmentationParams]]
    dataloader: Dict = field(default_factory=dict)

@dataclass
class DataloaderParams:
    # I think it must be list, with Enum Phase inside DataParams
    train: Optional[DataParams]
    valid: Optional[DataParams]
    test: Optional[DataParams]
    predict: Optional[DataParams]


# Losses parameters
@dataclass
class LossParams:
    name: str
    tag: Optional[str]
    mapping: Dict[str, str]
    weight: Optional[float]
    params: Dict = field(default_factory=dict)


# Metric parameters
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


# Config parameters
@dataclass
class ConfigParams:
    data: DataloaderParams
    optimization: List[OptimizationParams]
    losses: List[LossParams]
    metrics: List[MetricParams]
