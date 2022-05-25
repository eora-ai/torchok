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
    params: Optional[Dict] = field(default_factory=dict)
    
@dataclass
class SchedulerPLParams:
    # See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
    # for more information.
    interval: Optional[str] = None
    frequency: Optional[int] = None
    monitor: Optional[str] = None
    strict: Optional[bool] = None
    name: Optional[str] = None

@dataclass
class SchedulerParams:
    name: str
    params: Optional[Dict] = field(default_factory=dict)
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
    params: Dict
    transform: List[AugmentationParams]
    augment: Optional[List[AugmentationParams]] = field(default_factory=list)

@dataclass
class DataParams:
    dataset: DatasetParams
    dataloader: Dict


# Losses parameters
@dataclass
class LossParams:
    name: str
    mapping: Dict[str, str]
    params: Optional[Dict] = field(default_factory=dict)
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
    params: Optional[Dict] = field(default_factory=dict)
    phases: Optional[List[Phase]] = field(default_factory=lambda: [Phase.TRAIN, Phase.VALID, Phase.TEST, Phase.PREDICT])
    prefix: Optional[str] = None


# Task parameters
@dataclass
class TaskParams:
    name: str
    params: Optional[Dict] = field(default_factory=dict)


# Trainer parameters
# TODO: add all Trainer parameters
@dataclass
class TrainerParams:
    max_epochs: int
    precision: int = 32
    gpus: Optional[List] = field(default_factory=list)


# Checkpoint parameters
@dataclass
class CheckpointParams:
    filename: Optional[str] = None
    monitor: str = 'valid/loss'
    verbose: bool = False
    save_last: bool = False
    save_top_k: Optional[int] = 1
    save_weights_only: bool = False
    mode: str = 'min'
    auto_insert_metric_name: bool = False
    every_n_train_steps: Optional[int] = None
    every_n_val_epochs: Optional[int] = None


# Config parameters
@dataclass
class ConfigParams:
    # TODO add Logger params
    task: TaskParams
    data: Dict[Phase, List[DataParams]]
    optimization: List[OptimizationParams]
    joint_loss: JointLossParams
    trainer: TrainerParams
    checkpoint: CheckpointParams
    experiment_name: str
    log_dir: str = './logs'
    job_link: str = 'local'
    metrics: Optional[List[MetricParams]] = field(default_factory=list)
    