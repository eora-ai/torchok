from weakref import proxy

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class ModelCheckpointWithOnnx(ModelCheckpoint):
    """A class checkpointing ckpt and onnx format."""
    FILE_EXTENSION = ''
    CKPT_EXTENSION = '.ckpt'
    ONNX_EXTENSION = '.onnx'

    def __init__(self, *args, to_save = True, onnx_params = None, **kwargs):
        """Init ModelCheckpointWithOnnx."""
        super().__init__(*args, **kwargs)
        self.onnx_params = onnx_params if onnx_params is not None else {}
        self.to_save = to_save

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """Override _save_checkpoint."""
        if self.to_save:
            input_tensors = trainer.model.input_tensors
            trainer.model.to_onnx(filepath + self.ONNX_EXTENSION, (*input_tensors,), **self.onnx_params)

        trainer.save_checkpoint(filepath + self.CKPT_EXTENSION, self.save_weights_only)
        self._last_global_step_saved = trainer.global_step

        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
