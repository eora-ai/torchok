import os
from weakref import proxy
from typing import Dict, Optional

from pytorch_lightning.utilities.types import _METRIC
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class ModelCheckpointWithOnnx(ModelCheckpoint):
    """A class checkpointing ckpt and onnx format."""
    CKPT_EXTENSION = '.ckpt'
    ONNX_EXTENSION = '.onnx'

    def __init__(self, *args, onnx_to_save=False, onnx_params=None, **kwargs):
        """Init ModelCheckpointWithOnnx."""
        super().__init__(*args, **kwargs)
        self.onnx_params = onnx_params if onnx_params is not None else {}
        self.onnx_to_save = onnx_to_save

    def format_checkpoint_name(
        self, metrics: Dict[str, _METRIC], filename: Optional[str] = None, ver: Optional[int] = None
    ) -> str:
        """Override format_checkpoint_name."""
        filename = filename or self.filename
        filename = self._format_checkpoint_name(filename, metrics, auto_insert_metric_name=self.auto_insert_metric_name)

        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))

        return os.path.join(self.dirpath, filename) if self.dirpath else filename

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Override _save_checkpoint."""
        trainer.save_checkpoint(filepath + self.CKPT_EXTENSION, self.save_weights_only)
        self._last_global_step_saved = trainer.global_step

        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

        if self.onnx_to_save:
            input_tensors = trainer.model.input_tensors
            trainer.model.to_onnx(filepath + self.ONNX_EXTENSION, (*input_tensors,), **self.onnx_params)
