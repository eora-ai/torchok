from omegaconf import DictConfig, ListConfig
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


# Phase utils
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
    interval: Optional[str] = None
    frequency: Optional[int] = None
    monitor: Optional[str] = None
    strict: Optional[bool] = None
    name: Optional[str] = None

@dataclass
class SchedulerParams:
    name: str
    params: Dict = field(default_factory=dict)
    pl_params: Optional[SchedulerPLParams] = None

@dataclass
class OptimizationParams:
    optimizer: OptmizerParams
    scheduler: Optional[SchedulerParams] = None


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
class DataLoaderParams:
    dataset: DatasetParams
    dataloader: Dict = field(default_factory=dict)

@dataclass
class DataParams:
    # I think it must be list, with Enum Phase inside DataParams
    train: Optional[List[DataLoaderParams]] = None
    valid: Optional[List[DataLoaderParams]] = None
    test: Optional[List[DataLoaderParams]] = None
    predict: Optional[List[DataLoaderParams]] = None


# Losses parameters
@dataclass
class LossParams:
    name: str
    mapping: Dict[str, str]
    params: Dict = field(default_factory=dict)
    tag: Optional[str] = None
    weight: Optional[float] = None

@dataclass
class JointLossParams:
    losses: List[LossParams]
    normalize_weights: bool = True


# Metric parameters
@dataclass
class MetricParams:
    name: str
    mapping: Dict[str, str]
    params: Dict = field(default_factory=dict)
    # Need Any type because if it Phase - it need be one of [TRAIN, VALID, TEST, PREDICT] to Enum convert
    # Overwise if it is a str it convert our Enum Phase to string and it will be 'Phase.TRAIN'
    phases: Optional[List[Phase]] = field(default_factory=list)
    prefix: Optional[str] = None


# Config parameters
@dataclass
class ConfigParams:
    data: DataParams
    optimization: List[OptimizationParams]
    joint_loss: JointLossParams
    metrics: Optional[List[MetricParams]] = field(default_factory=list)

    def __post_init__(self):
        """Post process for data phases. 
        
        Convert string phase keys to enum.
        Hydra can't call __post_init__ of it's fields, because it's actually OmegaConf dict or list.
        """
        phase_mapping = {phase.value: phase for phase in Phase}

        # Change dataloaders phase keys to Enum
        data_with_enum = {}
        for key, value in self.data.items():
            phase_enum = phase_mapping[key]
            data_with_enum[phase_enum] = value

        self.data = DictConfig(data_with_enum)
