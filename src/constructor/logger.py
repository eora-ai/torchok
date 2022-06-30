import glob
import json
import boto3
from pathlib import Path
from typing import Any, Dict, Optional
from pytorch_lightning.loggers import mlflow, tensorboard
from pytorch_lightning.loggers.mlflow import rank_zero_only

from src.constructor.config_structure import LoggerType


class MLFlowLogger(mlflow.MLFlowLogger):
    """
    Log using `MLflow <https://mlflow.org>`_.
    Install it with pip:
    .. code-block:: bash
        pip install mlflow
    .. code-block:: python
        from pytorch_lightning import Trainer
        from pytorch_lightning.loggers import MLFlowLogger
        mlf_logger = MLFlowLogger(
            experiment_name="default",
            tracking_uri="file:./ml-runs"
        )
        trainer = Trainer(logger=mlf_logger)
    Use the logger anywhere in your :class:`~pytorch_lightning.core.lightning.LightningModule` as follows:
    .. code-block:: python
        from pytorch_lightning import LightningModule
        class LitModel(LightningModule):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_ml_flow_supports(...)
            def any_lightning_module_function_or_hook(self):
                self.logger.experiment.whatever_ml_flow_supports(...)
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
        super(mlflow.MLFlowLogger, self).finalize(status)
        status = 'FINISHED' if status == 'success' else status

        base_dir = str(self._save_dir)
        upload_file_paths = glob.glob(base_dir + '/*.ckpt') + glob.glob(base_dir + '/*.onnx')
        for file_path in upload_file_paths:
            self.experiment.log_artifact(self._run_id, file_path)

        if self.experiment.get_run(self._run_id):
            self.experiment.set_terminated(self._run_id, status)


def create_mlflow_logger(logger_params, outputs_path, experiment_name, version, job_link):
    save_dir = outputs_path / experiment_name / version
    remote_save_dir = logger_params.save_dir

    secretsmanager = logger_params.secrets_manager
    if secretsmanager:
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=secretsmanager.region)
        mlflow_secret = client.get_secret_value(SecretId=secretsmanager.mlflow_secret)
        mlflowdb_conf = json.loads(mlflow_secret["SecretString"])
        tracking_uri = (f"mysql+pymysql://{mlflowdb_conf['username']}:"
                        f"{mlflowdb_conf['password']}@{mlflowdb_conf['host']}/mlflow")
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


def create_logger(logger_params, outputs_path, experiment_name, experiment_subdir, job_link):
    if logger_params.logger == LoggerType.TENSORBOARD:
        return tensorboard.TensorBoardLogger(outputs_path, experiment_name, experiment_subdir, **logger_params.params)
    else:
        return create_mlflow_logger(logger_params.params, outputs_path, experiment_name, experiment_subdir, job_link)
    