from pathlib import Path
from argparse import Namespace
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:  # pragma: no-cover
    mlflow = None
    MlflowClient = None

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers import mlflow, tensorboard
from pytorch_lightning.loggers.mlflow import rank_zero_only, rank_zero_experiment


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
        for checkpoint in self._save_dir.rglob('*.ckpt'):
            self.experiment.log_artifact(self._run_id, str(checkpoint))
        if self.experiment.get_run(self._run_id):
            self.experiment.set_terminated(self._run_id, status)

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        def _log_recursive(params, prefix=''):
            for key, val in params.items():
                if isinstance(val, dict):
                    _log_recursive(val, prefix + f'{key}.')
                elif isinstance(val, BaseModel):
                    _log_recursive(val.dict(), prefix + f'{key}.')
                else:
                    self.experiment.log_param(self.run_id, prefix + f'{key}', val)

        _log_recursive(dict(params))


class TensorBoardLogger(tensorboard.TensorBoardLogger):
    r"""
    Log to local file system in `TensorBoard <https://www.tensorflow.org/tensorboard>`_ format.
    Implemented using :class:`~torch.utils.tensorboard.SummaryWriter`. Logs are saved to
    ``os.path.join(save_dir, name, version)``. This is the default logger in Lightning, it comes
    preinstalled.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import TensorBoardLogger
        >>> logger = TensorBoardLogger("tb_logs", name="my_model")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``. If it is the empty string then no per-experiment
            subdirectory is used.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used.
        log_graph: Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model.
        default_hp_metric: Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is
            called without a metric (otherwise calls to log_hyperparams without a metric are ignored).
        \**kwargs: Additional arguments like `comment`, `filename_suffix`, etc. used by
            :class:`SummaryWriter` can be passed as keyword arguments in this logger.

    """
    NAME_HPARAMS_FILE = 'hparams.yaml'

    def __init__(
        self,
        save_dir: str,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        **kwargs
    ):
        super().__init__(save_dir=save_dir, name=name, version=version, log_graph=log_graph,
                         default_hp_metric=default_hp_metric, **kwargs)

    @staticmethod
    def _convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
        # in case converting from namespace
        def convert(obj):
            if isinstance(obj, BaseModel):
                obj = obj.dict()

            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            else:
                return obj

        params = convert(params)

        if isinstance(params, Namespace):
            params = vars(params)

        if params is None:
            params = {}

        return params
