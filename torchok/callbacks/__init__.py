from pytorch_lightning.callbacks import (BackboneFinetuning, DeviceStatsMonitor, EarlyStopping,
                                         GradientAccumulationScheduler, LearningRateMonitor, ModelCheckpoint,
                                         ModelPruning, ModelSummary, QuantizationAwareTraining, RichModelSummary,
                                         RichProgressBar, StochasticWeightAveraging, Timer, TQDMProgressBar)

import torchok.callbacks.finalize_logger
import torchok.callbacks.freeze_unfreeze
import torchok.callbacks.checkpoint_onnx
from torchok.constructor import CALLBACKS

CALLBACKS.register_class(BackboneFinetuning)
CALLBACKS.register_class(DeviceStatsMonitor)
CALLBACKS.register_class(EarlyStopping)
CALLBACKS.register_class(GradientAccumulationScheduler)
CALLBACKS.register_class(LearningRateMonitor)
CALLBACKS.register_class(ModelPruning)
CALLBACKS.register_class(ModelSummary)
CALLBACKS.register_class(ModelCheckpoint)
CALLBACKS.register_class(RichModelSummary)
CALLBACKS.register_class(RichProgressBar)
CALLBACKS.register_class(StochasticWeightAveraging)
CALLBACKS.register_class(Timer)
CALLBACKS.register_class(TQDMProgressBar)
CALLBACKS.register_class(QuantizationAwareTraining)
