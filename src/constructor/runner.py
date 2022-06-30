import datetime
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.constructor.logger import create_logger
from src.callbacks.finalize_logger import FinalizeLogger


def create_outputs_path(log_dir: str, experiment_name: str, create_datetime_log_subdir: bool):
    if create_datetime_log_subdir:
        experiment_subdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        experiment_subdir = ''

    log_dir = Path(log_dir)
    full_outputs_path = log_dir / experiment_name / experiment_subdir
    full_outputs_path.mkdir(exist_ok=True, parents=True)

    return log_dir, experiment_name, experiment_subdir, full_outputs_path


def create_trainer(train_config):
    outputs_path, experiment_name, experiment_subdir,\
        full_outputs_path = create_outputs_path(log_dir=train_config.log_dir,
                                                experiment_name=train_config.experiment_name, 
                                                create_datetime_log_subdir=train_config.create_datetime_log_subdir)
    
    logger = create_logger(logger_params=train_config.logger,
                           outputs_path=outputs_path,
                           experiment_name=experiment_name,
                           experiment_subdir=experiment_subdir,
                           job_link=train_config.job_link)
                           
    checkpoint_callback = ModelCheckpoint(**train_config.checkpoint, dirpath=str(full_outputs_path))
    finalize_logger_callback = FinalizeLogger()
    callbacks = [checkpoint_callback, finalize_logger_callback]

    trainer = Trainer(logger=logger, callbacks=callbacks, **train_config.trainer)
    return trainer
    