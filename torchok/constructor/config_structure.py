from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# Phase utils
class Phase(Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    PREDICT = 'predict'


# Callbacks parameters
@dataclass
class CallbacksParams:
    name: str
    params: Optional[Dict] = field(default_factory=dict)


# Optimization parameters
@dataclass
class OptmizerParams:
    name: str
    params: Optional[Dict] = field(default_factory=dict)
    paramwise_cfg: Optional[Dict] = field(default_factory=dict)


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
    tag: Optional[str] = None


# Seed
@dataclass
class SeedParams:
    seed: Optional[int] = None
    workers: Optional[bool] = False


# Load checkpoint params
@dataclass
class LoadCheckpointParams:
    base_ckpt_path: Optional[str] = None
    overridden_name2ckpt_path: Optional[Dict[str, str]] = None
    exclude_keys: Optional[List[str]] = None
    strict: bool = True


# Task parameters
@dataclass
class TaskParams:
    name: str
    compute_loss_on_valid: bool = True
    params: Optional[Dict] = field(default_factory=dict)
    load_checkpoint: Optional[LoadCheckpointParams] = None


# Trainer parameters
@dataclass
class TrainerParams:
    enable_checkpointing: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    num_nodes: int = 1
    devices: Optional[Any] = None  # Union[List[int], str, int]
    auto_select_gpus: bool = False
    enable_progress_bar: bool = True
    overfit_batches: Any = 0.0  # Union[int, float]
    track_grad_norm: Any = -1  # Union[int, float, str]
    check_val_every_n_epoch: Optional[int] = 1
    fast_dev_run: Any = False  # Union[int, bool]
    accumulate_grad_batches: Optional[Any] = None  # Optional[Union[int, Dict[int, int]]]
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[Any] = None  # Union[str, timedelta, Dict[str, int]]
    limit_train_batches: Optional[Any] = None  # Optional[Union[int, float]]
    limit_val_batches: Optional[Any] = None  # Optional[Union[int, float]]
    limit_test_batches: Optional[Any] = None  # Optional[Union[int, float]]
    limit_predict_batches: Optional[Any] = None  # Optional[Union[int, float]]
    val_check_interval: Optional[Any] = None  # Optional[Union[int, float]]
    log_every_n_steps: int = 50
    accelerator: Optional[str] = None
    strategy: Optional[str] = None
    sync_batchnorm: bool = False
    precision: Any = 32  # Union[int, str]
    enable_model_summary: bool = True
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Optional[str] = None
    profiler: Optional[str] = None
    benchmark: Optional[bool] = None
    deterministic: Optional[bool] = None
    reload_dataloaders_every_n_epochs: int = 0
    auto_lr_find: bool = False
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: bool = False
    amp_backend: str = "native"
    amp_level: Optional[str] = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = "max_size_cycle"


# Logger
@dataclass
class LoggerParams:
    name: str
    log_dir: str
    experiment_name: str = 'default'
    timestamp: Optional[str] = None
    params: Optional[Dict] = field(default_factory=dict)


# Config parameters
@dataclass
class ConfigParams:
    task: TaskParams
    data: Dict[Phase, List[DataParams]]
    trainer: TrainerParams
    optimization: Optional[List[OptimizationParams]] = None
    joint_loss: Optional[JointLossParams] = None
    logger: Optional[LoggerParams] = None
    metrics: Optional[List[MetricParams]] = field(default_factory=list)
    callbacks: Optional[List[CallbacksParams]] = field(default_factory=list)
    resume_path: Optional[str] = None
    seed_params: Optional[SeedParams] = None
