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
    interval: Optional[str] = 'epoch'
    frequency: Optional[int] = 1
    monitor: Optional[str] = 'val_loss'
    strict: Optional[bool] = True
    name: Optional[str] = None

@dataclass
class SchedulerParams:
    name: str
    params: Optional[Dict] = field(default_factory=dict)
    pl_params: Optional[SchedulerPLParams] = field(default_factory=lambda: SchedulerPLParams())

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
@dataclass
class TrainerParams:
    logger: bool = True
    enable_checkpointing: bool = True
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = 'norm'
    num_nodes: int = 1
    num_processes: int = 1
    auto_select_gpus: bool = False
    enable_progress_bar: bool = False
    overfit_batches: float = 0.0
    track_grad_norm: int = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    max_steps: int = -1
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0
    limit_predict_batches: float = 1.0
    val_check_interval: float = 1.0
    log_every_n_steps: int = 50
    sync_batchnorm: bool = False
    precision: int = 32
    enable_model_summary: bool = True
    num_sanity_val_steps: int = 2
    deterministic: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    auto_lr_find: bool = False
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: bool = False
    amp_backend: str = 'native'
    amp_level: str = 'O2'
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = 'max_size_cycle'
    gpus: Optional[List[int]] = None
    default_root_dir: Optional[str] = None
    devices: Optional[List[int]] = None
    tpu_cores: Optional[List[int]] = None
    ipus: Optional[int] = None
    accumulate_grad_batches: Optional[Dict[int, int]] = None
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    min_steps: Optional[int] = None
    max_time: Optional[str] = None
    strategy: Optional[str] = None
    profiler: Optional[str] = None
    benchmark: Optional[bool] = None 


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
    