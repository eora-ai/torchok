from typing import Any, Dict, Tuple, List, Optional
from abc import ABC, abstractmethod

import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig

from src.constructor.config_structure import Phase
from src.constructor.constuctor import Constructor


class BaseTask(LightningModule, ABC):
    """An abstract class that represent main methods of tasks."""

    def __init__(self, hparams: DictConfig):
        # TODO: change type to ConfigParams
        """Init BaseTask.
        
        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.__constructor = Constructor(hparams)
        self._metrics_manager = self.__constructor.configure_metrics_manager()
        self._losses = self.__constructor.configure_losses()
        self._hparams = hparams
        self.__input_shapes = self._hparams.task.input_shapes
        self.__input_dtypes = self._hparams.task.input_dtypes
        self._input_tensors = []

        for input_shape, input_dtype in zip(self.__input_shapes, self.__input_dtypes):
            input_tensor = torch.rand(*input_shape).type(torch.__dict__[input_dtype])
            self._input_tensors.append(input_tensor)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Abstract forward method for validation an test."""
        pass

    @abstractmethod
    def forward_with_gt(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Abstract forward method for training(with ground truth labels)."""
        pass

    def on_train_start(self) -> None:
        # TODO check and load checkpoint
        pass

    def on_test_start(self) -> None:
        # TODO check and load checkpoint
        pass

    def training_epoch_end(self, outputs: Tuple[torch.Tensor, Dict]) -> None:
        """It's calling at the end of the training epoch with the outputs of all training steps."""
        total_loss, tagged_loss_values = outputs
        avg_loss = total_loss.mean()
        self.log('train/avg_loss', avg_loss, on_step=False, on_epoch=True)

        for tag, loss_value in tagged_loss_values.items():
            self.log(f'train/{tag}', loss_value, on_step=False, on_epoch=True)

        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        self.log_dict(self._metrics_manager.on_epoch_end(Phase.TRAIN))

    def validation_epoch_end(self, outputs: Tuple[torch.Tensor, Dict]) -> None:
        """It's calling at the end of the validation epoch with the outputs of all validation steps."""
        total_loss, tagged_loss_values = outputs
        avg_loss = total_loss.mean()
        self.log('valid/avg_loss', avg_loss, on_step=False, on_epoch=True)

        for tag, loss_value in tagged_loss_values.items():
            self.log(f'valid/{tag}', loss_value, on_step=False, on_epoch=True)

        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        self.log_dict(self._metrics_manager.on_epoch_end(Phase.VALID))

    def test_epoch_end(self, outputs: Tuple[torch.Tensor, Dict]) -> None:
        """It's calling at the end of a test epoch with the output of all test steps."""
        total_loss, tagged_loss_values = outputs
        avg_loss = total_loss.mean()
        self.log('test/avg_loss', avg_loss, on_step=False, on_epoch=True)

        for tag, loss_value in tagged_loss_values.items():
            self.log(f'test/{tag}', loss_value, on_step=False, on_epoch=True)

        self.log_dict(self._metrics_manager.on_epoch_end(Phase.TEST))

    def to_onnx(self, onnx_params) -> None:
        """It's saving the model in ONNX format."""
        super().to_onnx(input_sample=self._input_tensors,
                        **onnx_params)

    def on_train_end(self) -> None:
        """It's calling at the end of training before logger experiment is closed."""
        onnx_params = self._hparams.onnx_params
        if onnx_params is not None:
            self.to_onnx(onnx_params)

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configure optimizers.

        Returns:
            This method return two lists.
            First list - optimizers.
            Second list - schedulers(elements can be None type).
        """
        configure_optimizers = self.__constructor.configure_optimizers(self.parameters())
        optimizers, schedulers = [], []

        for item in configure_optimizers:
            optimizers.append(item['optimizer'])
            lr_scheduler = item.get('lr_scheduler')
            if lr_scheduler is not None:
                schedulers.append(lr_scheduler['scheduler'])
            else:
                schedulers.append(None)
        return optimizers, schedulers

    def train_dataloader(self) -> Optional[List[torch.DataLoader]]:
        """Implement one or more PyTorch DataLoaders for training."""
        data_params = self._hparams['data'].get(Phase.TRAIN, None)

        if data_params is None:
            return None

        data_loader = self.__constructor.create_dataloaders(Phase.TRAIN)
        return data_loader

    def val_dataloader(self) -> Optional[List[torch.DataLoader]]:
        """Implement one or multiple PyTorch DataLoaders for prediction."""
        data_params = self._hparams['data'].get(Phase.VALID, None)

        if data_params is None:
            return None

        for data_param in data_params:
            drop_last = data_param['dataloader']['drop_last']
            if drop_last:
                raise ValueError(f'DataLoader parametrs `drop_last` must be False in {Phase.VALID.value} phase.')

        data_loader = self.__constructor.create_dataloaders(Phase.VALID)
        return data_loader

    def test_dataloader(self) -> Optional[List[torch.DataLoader]]:
        """Implement one or multiple PyTorch DataLoaders for testing."""
        data_params = self._hparams['data'].get(Phase.TEST, None)

        if data_params is None:
            return None

        for data_param in data_params:
            drop_last = data_param['dataloader']['drop_last']
            if drop_last:
                raise ValueError(f'DataLoader parametrs `drop_last` must be False in {Phase.TEST.value} phase.')

        data_loader = self.__constructor.create_dataloaders(Phase.TEST)
        return data_loader

    def predict_dataloader(self) -> Optional[List[torch.DataLoader]]:
        """Implement one or multiple PyTorch DataLoaders for prediction."""
        data_params = self._hparams['data'].get(Phase.PREDICT, None)

        if data_params is None:
            return None

        for data_param in data_params:
            drop_last = data_param['dataloader']['drop_last']
            if drop_last:
                raise ValueError(f'DataLoader parametrs `drop_last` must be False in {Phase.PREDICT.value} phase.')

        data_loader = self.__constructor.create_dataloaders(Phase.PREDICT)
        return data_loader

    def training_step_end(self, step_outputs: torch.Tensor) -> torch.Tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    def validation_step_end(self, step_outputs: torch.Tensor) -> torch.Tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    def test_step_end(self, step_outputs: torch.Tensor) -> torch.Tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    @property
    def hparams(self) -> DictConfig:
        """Hyperparameters that set in yaml file."""
        return self._hparams

    @property
    def metrics_manager(self):
        """Metrics manager."""
        return self._metrics_manager

    @property
    def losses(self):
        """Losses."""
        return self._losses

    @property
    def input_shapes(self) -> List[Tuple[int, int, int]]:
        """Input shapes."""
        return self.__input_shapes

    @property
    def input_dtypes(self) -> List[str]:
        """Input dtypes."""
        return self.__input_dtypes

    @property
    def input_tensors(self) -> List[torch.Tensor]:
        """Input tensors."""
        return self._input_tensors
