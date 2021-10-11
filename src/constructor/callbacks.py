from pydantic import BaseModel
from pytorch_lightning.callbacks import EarlyStopping, GPUStatsMonitor, LearningRateMonitor, Timer

from src.registry import CALLBACKS

CALLBACKS.register_class(Timer)
CALLBACKS.register_class(EarlyStopping)
CALLBACKS.register_class(GPUStatsMonitor)
CALLBACKS.register_class(LearningRateMonitor)


class TimerParams(BaseModel):
    duration: str = None
    interval: str = 'step'
    verbose: bool = True


class EarlyStoppingParams(BaseModel):
    monitor: str = 'valid/loss'
    min_delta: float = 0.0
    patience: int = 3
    verbose: bool = False
    mode: str = 'min'
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: float = None
    divergence_threshold: float = None
    check_on_train_epoch_end: bool = False


class GPUStatsMonitorParams(BaseModel):
    memory_utilization: bool = True
    gpu_utilization: bool = True
    intra_step_time: bool = False
    inter_step_time: bool = False
    fan_speed: bool = False
    temperature: bool = False


class LearningRateMonitorParams(BaseModel):
    logging_interval: str = None
    log_momentum: bool = False


def create_callbacks(callbacks):
    return [CALLBACKS[cb.name](**cb.params) for cb in callbacks]
