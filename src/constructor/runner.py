from pytorch_lightning import Trainer
from pathlib import Path

from src.constructor.logger import create_logger, create_outputs_path
from src.callbacks.finalize_logger import FinalizeLogger
from src.callbacks.model_checkpoint_with_onnx import ModelCheckpointWithOnnx


def create_trainer(train_config):
    logger_params = train_config.logger
    if logger_params is not None:
        full_outputs_path = create_outputs_path(log_dir=logger_params.log_dir,
                                                experiment_name=logger_params.experiment_name,
                                                create_datetime_log_subdir=logger_params.create_datetime_log_subdir)

        experiment_path = Path(logger_params.log_dir) / logger_params.experiment_name
        experiment_subdir = str(full_outputs_path.relative_to(experiment_path))

        logger = create_logger(logger_class_name=logger_params.name,
                               logger_class_params=logger_params.params,
                               outputs_path=logger_params.log_dir,
                               experiment_name=logger_params.experiment_name,
                               experiment_subdir=experiment_subdir,
                               full_outputs_path=full_outputs_path)

        if train_config.checkpoint is not None:
            checkpoint_callback = ModelCheckpointWithOnnx(**train_config.checkpoint, dirpath=str(full_outputs_path))
            finalize_logger_callback = FinalizeLogger()
            callbacks = [checkpoint_callback, finalize_logger_callback]
        else:
            callbacks = None
    else:
        logger = True
        callbacks = None

    trainer = Trainer(logger=logger, callbacks=callbacks, **train_config.trainer)
    return trainer
