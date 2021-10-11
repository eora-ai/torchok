import datetime
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import torch
from pymysql import converters
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler, PyTorchProfiler

from .callbacks import create_callbacks
from .config_structure import TensorboardLoggerParams, MLFlowLoggerParams
from .loggers import MLFlowLogger, TensorBoardLogger


def create_logger(logger_params, outputs_path, experiment_name, version, job_link):
    if isinstance(logger_params, TensorboardLoggerParams):
        logger = create_tensorboard_logger(logger_params, outputs_path, experiment_name, version)
    elif isinstance(logger_params, MLFlowLoggerParams):
        logger = create_mlflow_logger(logger_params, outputs_path, experiment_name, version, job_link)
    else:
        raise ValueError(f'Unknown logger params: {type(logger_params)}')

    return logger


def create_tensorboard_logger(logger_params, outputs_path, experiment_name, version):
    params = dict(save_dir=str(outputs_path),
                  name=experiment_name,
                  version=version,
                  default_hp_metric=False)
    params.update(logger_params.dict())
    params.pop('logger')
    logger = TensorBoardLogger(**params)

    return logger


def create_mlflow_logger(logger_params, outputs_path, experiment_name, version, job_link):
    save_dir = outputs_path / experiment_name / version
    remote_save_dir = logger_params.save_dir
    converters.encoders[np.float64] = converters.escape_float
    converters.conversions = converters.encoders.copy()
    converters.conversions.update(converters.decoders)

    secretsmanager = logger_params.secrets_manager
    if secretsmanager:
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=secretsmanager.region)
        mlflow_secret = client.get_secret_value(SecretId=secretsmanager.mlflow_secret)
        mlflowdb_conf = json.loads(mlflow_secret["SecretString"])
        tracking_uri = f"mysql+pymysql://{mlflowdb_conf['username']}:{mlflowdb_conf['password']}@{mlflowdb_conf['host']}/mlflow"
    else:
        tracking_uri = logger_params.tracking_uri

    if job_link != 'local':
        logger_params.tags['mlflow.source.name'] = job_link
        logger_params.tags['mlflow.source.type'] = 'JOB'

    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=logger_params.tags['mlflow.runName'],
        tracking_uri=tracking_uri,
        tags=logger_params.tags,
        save_dir=str(save_dir),
        artifact_location=str(remote_save_dir)
    )

    return logger


def create_checkpoint_callback(checkpoint_params, dirpath):
    return ModelCheckpoint(**checkpoint_params.dict(), dirpath=str(dirpath))


def create_outputs_path(log_dir, experiment_name):
    version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(log_dir)
    full_outputs_path = log_dir / experiment_name / version
    full_outputs_path.mkdir(exist_ok=True, parents=True)

    return log_dir, experiment_name, version, full_outputs_path


def create_profiler(profiler_params, checkpoint_path):
    if profiler_params is None:
        return None
    else:
        if profiler_params.save_profile:
            output_filename = checkpoint_path / 'profile.log'
        else:
            output_filename = None

        if profiler_params.name == 'simple':
            return SimpleProfiler(output_filename)
        elif profiler_params.name == 'advanced':
            return AdvancedProfiler(output_filename)
        elif profiler_params.name == 'network':
            return PyTorchProfiler(output_filename)
        else:
            raise ValueError('Given type of profiler is not supported. Use `simple` or `advanced`')


def download_s3_artifact(s3_path, local_path):
    parsed = urlparse(s3_path, allow_fragments=False)
    bucket_name, prefix = parsed.netloc, parsed.path[1:]

    local_path = Path(local_path)
    if local_path.is_file():
        local_path = str(local_path)
    else:
        local_path = str(local_path / Path(prefix).name)

    session = boto3.session.Session()
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)
    bucket.download_file(prefix, local_path)

    return local_path


def restore_checkpoint(restore_path, dest_checkpoint_path, do_restore):
    if not restore_path:
        if do_restore:
            print('WARN: You wanted to restore from checkpoint but not specified restore_path. '
                  'Continue without restoring...')
        return None
    elif not do_restore:
        print('WARN: You specified restore_path but not set do_restore to True. Continue without restoring...')
        return None

    # s3 files
    if restore_path.startswith('s3'):
        print(f"Downloading checkpoint from {restore_path} to {dest_checkpoint_path}")
        dest_checkpoint_path = download_s3_artifact(restore_path, dest_checkpoint_path)

        return dest_checkpoint_path

    # local files
    path = Path(restore_path)
    if path.is_file() and path.suffix == '.ckpt':
        for filename in path.parent.glob('**/events*'):
            shutil.copy(filename, dest_checkpoint_path)
    elif path.is_dir():
        for filename in path.glob('**/events*'):
            shutil.copy(filename, dest_checkpoint_path)

        ckpt_paths = list(path.glob('**/last.ckpt'))
        if len(ckpt_paths) == 0:
            restore_path = None
            print(f'No checkpoints for resume found, continue from scratch...')
        elif len(ckpt_paths) == 1:
            restore_path = ckpt_paths[0].as_posix()
            print(f'Last checkpoint to resume from was found: {restore_path}')
        else:
            # Choose checkpoint with the highest global_step
            restore_path = max(ckpt_paths, key=lambda x: torch.load(x, map_location='cpu')['global_step'])
            restore_path = restore_path.as_posix()
            print(f'Last checkpoint to resume from was found: {restore_path}')
    else:
        raise ValueError(f'Invalid path to checkpoint: {path}')

    return restore_path


def create_trainer(train_config, job_link):
    outputs_path, experiment_name, version, full_outputs_path = create_outputs_path(train_config.log_dir,
                                                                                    train_config.experiment_name)
    logger = create_logger(train_config.logger, outputs_path, experiment_name, version, job_link)

    checkpoint_callback = create_checkpoint_callback(train_config.checkpoint, full_outputs_path)
    profiler = create_profiler(train_config.profiler, full_outputs_path)
    checkpoint_restore_path = restore_checkpoint(train_config.restore_path, full_outputs_path, train_config.do_restore)

    trainer_params = train_config.trainer.dict()

    callbacks = create_callbacks(train_config.callbacks)
    callbacks.append(checkpoint_callback)

    trainer = Trainer(logger=logger, profiler=profiler, callbacks=callbacks,
                      resume_from_checkpoint=checkpoint_restore_path,
                      **trainer_params)
    return trainer
