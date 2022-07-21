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
    enable_checkpointing: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[str] = None
    num_nodes: int = 1
    num_processes: Optional[int] = None  # TODO: Remove in 2.0
    devices: Optional[List[int]] = None
    gpus: Optional[List[int]] = None
    auto_select_gpus: bool = False
    tpu_cores: Optional[List[int]] = None  # TODO: Remove in 2.0
    ipus: Optional[int] = None  # TODO: Remove in 2.0
    enable_progress_bar: bool = True
    overfit_batches: float = 0.0
    track_grad_norm: float = -1
    check_val_every_n_epoch: Optional[int] = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: Optional[Dict[int, int]] = None
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    max_steps: int = -1
    min_steps: Optional[int] = None
    max_time: Optional[Dict[str, int]] = None
    limit_train_batches: Optional[float] = None
    limit_val_batches: Optional[float] = None
    limit_test_batches: Optional[float] = None
    limit_predict_batches: Optional[float] = None
    val_check_interval: Optional[float] = None
    log_every_n_steps: int = 50
    accelerator: Optional[str] = None
    strategy: Optional[str] = None
    sync_batchnorm: bool = False
    precision: int = 32
    enable_model_summary: bool = True
    weights_save_path: Optional[str] = None  # TODO: Remove in 1.8
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Optional[str] = None
    profiler: Optional[str] = None
    benchmark: Optional[bool] = None
    reload_dataloaders_every_n_epochs: int = 0
    auto_lr_find: bool = False
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: bool = False
    amp_backend: str = "native"
    amp_level: Optional[str] = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = "max_size_cycle"


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
    export_to_onnx: bool = False
    onnx_params: Dict = field(default_factory=dict)


# Config parameters
@dataclass
class ConfigParams:
    # TODO add Logger params
    task: TaskParams
    data: Dict[Phase, List[DataParams]]
    trainer: TrainerParams
    checkpoint: CheckpointParams
    experiment_name: str
    log_dir: str = './logs'
    optimization: Optional[List[OptimizationParams]] = field(default_factory=list)
    joint_loss: Optional[JointLossParams] = None
    metrics: Optional[List[MetricParams]] = field(default_factory=list)
