from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


def create_trainer(train_config, metadata, job_link):

    logger = TensorBoardLogger(metadata['log_dir'], metadata['experiment_name'], metadata['version'])
    checkpoint_callback = ModelCheckpoint(**train_config.checkpoint, dirpath=str(metadata['full_outputs_path']))
    callbacks = [checkpoint_callback]
    trainer = Trainer(logger=logger, callbacks=callbacks, **train_config.trainer)
    return trainer
