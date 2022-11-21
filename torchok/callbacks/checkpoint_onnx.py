from copy import deepcopy
from typing import Dict
from weakref import proxy

import torch
from torch import Tensor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.trainer import Trainer

from torchok.constructor import CALLBACKS


@CALLBACKS.register_class
class CheckpointONNX(ModelCheckpoint):
    """A class checkpointing onnx format."""
    ONNX_EXTENSION = '.onnx'

    def __init__(self, *args, onnx_params=None, remove_head=False, **kwargs):
        """Init CheckpointONNX."""
        super().__init__(*args, **kwargs)
        self.onnx_params = onnx_params if onnx_params is not None else {}
        self.remove_head = remove_head

    def _update_best_and_save(
        self, current: Tensor, trainer: Trainer, monitor_candidates: Dict[str, Tensor]
    ) -> None:
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

        # save the current score
        self.current_score = current
        self.best_k_models[filepath] = current

        if len(self.best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
        self._save_checkpoint(trainer, filepath)

        if del_filepath is not None and filepath != del_filepath:
            onnx_del_path = del_filepath.replace(self.FILE_EXTENSION, self.ONNX_EXTENSION)
            trainer.strategy.remove_checkpoint(onnx_del_path)

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        """Override _save_checkpoint."""
        self._last_global_step_saved = trainer.global_step
        if trainer.is_global_zero:
            # DDP mode use some wrappers, and we go down to BaseModel.
            model = trainer.model.module.module if trainer.num_devices > 1 else trainer.model
            input_tensors = [getattr(model, name) for name in model.input_tensor_names]
            model = model.as_module()
            if self.remove_head:
                model = model[:-1]
            model = deepcopy(model)
            onnx_file_path = filepath.replace(self.FILE_EXTENSION, self.ONNX_EXTENSION)
            torch.onnx.export(model, tuple(input_tensors), onnx_file_path, **self.onnx_params)

            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
