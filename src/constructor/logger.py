import datetime
from pathlib import Path
from itertools import chain
from typing import Any, Dict, Optional, Union
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.mlflow import rank_zero_only


def create_outputs_path(log_dir: str, experiment_name: str, create_datetime_log_subdir: bool) -> Path:
    """Create directory for saving checkpoints and logging metrics.

    Args:
        log_dir: Base path.
        experiment_name: Sub directory for log_dir.
        create_datetime_log_subdir: Whether to create log_dir/experiment_name/%Y-%m-%d_%H-%M-%S or
            log_dir/experiment_name/ folders. This datetime would be some kind a version of experiment.

    Returns:
        full_outputs_path: Directory path to save checkpoints and logging metrics.
    """
    if create_datetime_log_subdir:
        experiment_subdir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        experiment_subdir = ''

    log_dir = Path(log_dir)
    full_outputs_path = log_dir / experiment_name / experiment_subdir
    full_outputs_path.mkdir(exist_ok=True, parents=True)

    return full_outputs_path


class MLFlowLoggerX(MLFlowLogger):
    """This logger completely repeats the functionality of Pytorch Lightning MLFlowLogger. But unlike the Lightning
    logger it upload *.onnx and *.ckpt artifacts to artifact_location path.

    Args:
        experiment_name: The name of the experiment
        tracking_uri: Address of local or remote tracking server.
            If not provided, defaults to `file:<save_dir>`.
        tags: A dictionary tags for the experiment.
        save_dir: A path to a local directory where the MLflow runs get saved.
            Defaults to `./mlflow` if `tracking_uri` is not provided.
            Has no effect if `tracking_uri` is provided.
        prefix: A string to put at the beginning of metric keys.
        artifact_location: The location to store run artifacts. If not provided, the server picks an appropriate
            default.

    Raises:
        ImportError:
            If required MLFlow package is not installed on the device.
    """
    def __init__(
            self,
            experiment_name: str = 'default',
            run_name: Optional[str] = None,
            tracking_uri: Optional[str] = None,
            tags: Optional[Dict[str, Any]] = None,
            save_dir: Optional[str] = './mlruns',
            prefix: str = '',
            artifact_location: Optional[str] = None
    ):
        super().__init__(experiment_name=experiment_name, run_name=run_name, tracking_uri=tracking_uri,
                         tags=tags, save_dir=save_dir, prefix=prefix, artifact_location=artifact_location)
        self._save_dir = Path(save_dir)

    @rank_zero_only
    def finalize(self, status: str = 'FINISHED') -> None:
        """Call finalize of pytorch lightning MlFlowLogger and logs *.ckpt and *.onnx artifacts in artifact_location."""
        super().finalize(status)
        status = 'FINISHED' if status == 'success' else status

        upload_file_paths = chain(self._save_dir.glob('*.ckpt'), self._save_dir.glob('*.onnx'))
        for file_path in upload_file_paths:
            self.experiment.log_artifact(self._run_id, file_path)

        if self.experiment.get_run(self._run_id):
            self.experiment.set_terminated(self._run_id, status)


def create_logger(logger_class_name: str, logger_class_params: Dict, outputs_path: Union[str, Path],
                  experiment_name: Union[str, Path], experiment_subdir: Union[str, Path],
                  full_outputs_path: Union[str, Path]) -> LightningLoggerBase:
    """Create logger.

    Args:
        logger_class_name: Logger class name.
        logger_class_params: Logger class constructor parameters.
        outputs_path: Base directory for parameters logging.
        experiment_name: Sub directory for output_path for parameters logging, the name means experiment name.
        experiment_subdir: Sub directory for experiment_name for parameters logging, the name means experiment
            start datetime, some kind of experiment version.
        full_outputs_path: Directory path for parameters logging.

    Raises:
        ValueError: If logger_class_name not in support logger names. Specifically TorchOk support next loggers:
            TensorBoardLogger, MlFlowLogger, MLFlowLoggerX, WandbLogger, CSVLogger, NeptuneLogger

    Returns:
        Logger.
    """
    if logger_class_name == 'TensorBoardLogger':
        logger_params = {
            'save_dir': outputs_path,
            'name': experiment_name,
            'version': experiment_subdir
        }
        logger_params.update(logger_class_params)
        return TensorBoardLogger(**logger_params)
    elif logger_class_name == 'MLFlowLoggerX':
        logger_params = {
            'save_dir': full_outputs_path,
            'experiment_name': experiment_name
        }
        logger_params.update(logger_class_params)
        return MLFlowLoggerX(**logger_params)
    elif logger_class_name == 'MLFlowLogger':
        logger_params = {
            'save_dir': full_outputs_path,
            'experiment_name': experiment_name
        }
        logger_params.update(logger_class_params)
        return MLFlowLogger(**logger_params)
    elif logger_class_name == 'WandbLogger':
        logger_params = {
            'save_dir': full_outputs_path,
            'name': experiment_name,
            'version': experiment_subdir
        }
        logger_params.update(logger_class_params)
        return WandbLogger(**logger_params)
    elif logger_class_name == 'CSVLogger':
        logger_params = {
            'save_dir': full_outputs_path,
            'name': experiment_name,
            'version': experiment_subdir
        }
        logger_params.update(logger_class_params)
        return CSVLogger(**logger_params)
    elif logger_class_name == 'NeptuneLogger':
        logger_params = {
            'project': full_outputs_path,
            'name': experiment_name,
        }
        logger_params.update(logger_class_params)
        return NeptuneLogger(**logger_params)
    else:
        raise ValueError(f'Create Logger method. TorchOk not support logger with name {logger_class_name}. '
                         f'TorchOk supports the following names: TensorBoardLogger, MlFlowLogger, MLFlowLoggerX, '
                         f'WandbLogger, CSVLogger, NeptuneLogger.')
