from pytorch_lightning import Trainer
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from src.callbacks.model_checkpoint_with_onnx import ModelCheckpointWithOnnx

def create_trainer(train_config, metadata, job_link):

    logger = TensorBoardLogger(metadata['log_dir'], metadata['experiment_name'], metadata['version'])
    checkpoint_with_onnx_callback = ModelCheckpointWithOnnx(**train_config.checkpoint, dirpath=str(metadata['full_outputs_path']))
    callbacks = [checkpoint_with_onnx_callback]
    trainer = Trainer(logger=logger, callbacks=callbacks, **train_config.trainer)
    return trainer
