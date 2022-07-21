import datetime
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from torchok.callbacks.model_checkpoint_with_onnx import ModelCheckpointWithOnnx


def create_outputs_path(log_dir, experiment_name):
    version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(log_dir)
    full_outputs_path = log_dir / experiment_name / version
    full_outputs_path.mkdir(exist_ok=True, parents=True)

    return log_dir, experiment_name, version, full_outputs_path


def create_trainer(train_config):
    outputs_path, experiment_name, version, full_outputs_path = create_outputs_path(train_config.log_dir,
                                                                                    train_config.experiment_name)
    logger = TensorBoardLogger(outputs_path, experiment_name, version)
    checkpoint_callback = ModelCheckpointWithOnnx(**train_config.checkpoint, dirpath=str(full_outputs_path))
    callbacks = [checkpoint_callback]
    trainer = Trainer(logger=logger, callbacks=callbacks, **train_config.trainer)
    return trainer
