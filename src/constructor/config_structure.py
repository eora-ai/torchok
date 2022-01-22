from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from pydantic import BaseModel, Field
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.plugins import CheckpointIO, PrecisionPlugin, TrainingTypePlugin
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.profiler import BaseProfiler
from typing_extensions import Literal


class StructureParams(BaseModel):
    name: str
    params: dict = {}
    aux_params: Optional[dict] = {}


class DataLoaderParams(BaseModel):
    batch_size: int
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = False
    use_custom_collate_fn: bool = False
    use_custom_batch_sampler: bool = False


class DatasetParams(BaseModel):
    name: str
    params: dict = {}
    transform: List[StructureParams]
    augment: List[StructureParams] = None
    dataloader_params: DataLoaderParams


class DataParams(BaseModel):
    common_params: Optional[dict] = {}
    train_params: Optional[DatasetParams]
    valid_params: Optional[DatasetParams]
    test_params: Optional[DatasetParams]


class TrainerParams(BaseModel):
    accelerator: Union[str, Accelerator, None] = None
    accumulate_grad_batches: Union[int, Dict[int, int], None] = None
    amp_backend: str = 'native'
    amp_level: Optional[str] = None
    auto_lr_find: Union[bool, str] = False
    auto_scale_batch_size: Union[str, bool] = False
    auto_select_gpus: bool = False
    benchmark: bool = False
    check_val_every_n_epoch: int = 1
    checkpoint_callback: Optional[bool] = None
    default_root_dir: Optional[str] = None
    detect_anomaly: bool = False
    deterministic: bool = False
    devices: Union[int, str, List[int], None] = None
    enable_model_summary: bool = True
    enable_progress_bar: bool = True
    fast_dev_run: Union[int, bool] = False
    flush_logs_every_n_steps: Optional[int] = None
    gpus: Union[int, str, List[int], None] = None
    gradient_clip_algorithm: Optional[str] = None
    gradient_clip_val: Union[int, float, None] = None
    ipus: Optional[int] = None
    limit_predict_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    log_every_n_steps: int = 50
    log_gpu_memory: Optional[str] = None
    max_epochs: Optional[int] = None
    max_steps: int = - 1
    max_time: Union[str, timedelta, Dict[str, int], None] = None
    min_epochs: Optional[int] = None
    min_steps: Optional[int] = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = 'max_size_cycle'
    num_nodes: int = 1
    num_processes: int = 1
    num_sanity_val_steps: int = 2
    overfit_batches: Union[int, float] = 0.0
    plugins: Union[TrainingTypePlugin, PrecisionPlugin, ClusterEnvironment, CheckpointIO, str, List[
        Union[TrainingTypePlugin, PrecisionPlugin, ClusterEnvironment, CheckpointIO, str]], None] = None
    precision: Union[int, str] = 32
    prepare_data_per_node: Optional[bool] = None
    process_position: int = 0
    progress_bar_refresh_rate: Optional[int] = None
    reload_dataloaders_every_n_epochs: int = 0
    replace_sampler_ddp: bool = True
    strategy: Union[str, TrainingTypePlugin, None] = None
    sync_batchnorm: bool = False
    tpu_cores: Union[int, str, List[int], None] = None
    track_grad_norm: Union[int, float, str] = - 1
    val_check_interval: Union[int, float] = 1.0
    weights_save_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class TensorboardLoggerParams(BaseModel):
    logger: Literal['tensorboard']
    log_graph: bool = False


class AWSSecretsManagerParams(BaseModel):
    region: str
    mlflow_secret: str


class MLFlowLoggerParams(BaseModel):
    logger: Literal['mlflow']
    tracking_uri: Optional[str] = None
    tags: Optional[Dict[str, Any]] = {}
    save_dir: Optional[str] = './mlruns'
    secrets_manager: Optional[AWSSecretsManagerParams] = None


class CheckpointParams(BaseModel):
    filename: Optional[str] = None
    monitor: str = 'valid/loss'
    verbose: bool = False
    save_last: bool = False
    save_top_k: Optional[int] = 1
    save_weights_only: bool = False
    mode: str = "min"
    auto_insert_metric_name: bool = False
    every_n_train_steps: Optional[int] = None
    every_n_val_epochs: Optional[int] = None


class ProfilerParams(BaseModel):
    name: str = 'simple'
    save_profile: bool = False


class LossParams(BaseModel):
    loss_list: List[StructureParams]
    weights: List[float] = None
    log_separate_losses: bool = False


class MetricParams(BaseModel):
    name: str
    params: dict = {}
    phases: List[Union[Literal['train'], Literal['valid'], Literal['test']]] = ['train', 'valid', 'test']


class TrainConfigParams(BaseModel, Namespace):
    task: StructureParams
    restore_path: Optional[str] = None
    do_restore: Optional[str] = None
    loss: LossParams
    optimizers: Union[StructureParams, Dict[str, StructureParams]]
    schedulers: Union[StructureParams, Dict[str, StructureParams]] = None
    data: DataParams
    metrics: List[MetricParams] = []
    trainer: TrainerParams
    logger: Union[TensorboardLoggerParams, MLFlowLoggerParams] = \
        Field(..., descriminator='logger')
    experiment_name: str
    log_dir: str = './logs'
    checkpoint: CheckpointParams
    callbacks: Optional[List[StructureParams]] = []
    profiler: Optional[ProfilerParams]
