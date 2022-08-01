from . import model_checkpoint_with_onnx
from . import finalize_logger
from . import freeze_unfreeze

from pytorch_lightning.callbacks import (
    DeviceStatsMonitor, EarlyStopping, GradientAccumulationScheduler, LearningRateMonitor, ModelPruning,
    ModelSummary, QuantizationAwareTraining, RichModelSummary, RichProgressBar, StochasticWeightAveraging,
    Timer, TQDMProgressBar, BackboneFinetuning
)

from torchok.constructor import CALLBACKS


CALLBACKS.register_class(DeviceStatsMonitor)
CALLBACKS.register_class(EarlyStopping)
CALLBACKS.register_class(GradientAccumulationScheduler)
CALLBACKS.register_class(LearningRateMonitor)
CALLBACKS.register_class(ModelPruning)
CALLBACKS.register_class(ModelSummary)
CALLBACKS.register_class(QuantizationAwareTraining)
CALLBACKS.register_class(RichModelSummary)
CALLBACKS.register_class(RichProgressBar)
CALLBACKS.register_class(StochasticWeightAveraging)
CALLBACKS.register_class(Timer)
CALLBACKS.register_class(TQDMProgressBar)
CALLBACKS.register_class(BackboneFinetuning)
