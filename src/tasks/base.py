from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig


from src.constructor.constuctor import Constructor


class BaseTask(LightningModule, ABC):
    """An abstract class that represent main methods of tasks."""

    def __init__(self, hparams: DictConfig):
        """Init BaseTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.__constructor = Constructor(hparams)
        self._metric_manager = self.__constructor.configure_metrics_manager()
        self._criterion = self.__constructor.configure_losses()
        self.__hparams = self.__constructor.hparams
        self.__input_shapes = self.__hparams.input_shapes
        self.example_input_array = [torch.rand(1, *shape) for _, shape in self.__input_shapes.items()]

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward_with_gt(batch: dict) -> dict:
        """Abstract method for forward with ground truth labels."""
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def validation_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def test_step(self, *args, **kwargs):
        pass

    def on_train_start(self) -> None:
        # TODO check and load checkpoint
        pass

    def on_test_start(self) -> None:
        # TODO check and load checkpoint
        pass

    def training_epoch_end(self, outputs: Tuple[torch.tensor, dict]) -> None:
        """It's calling at the end of the training epoch with the outputs of all training steps."""
        total_loss, tagged_loss_values = outputs
        avg_loss = total_loss.mean()
        self.log('train/avg_loss', avg_loss, on_step=False, on_epoch=True)

        for tag, loss_value in tagged_loss_values.items():
            self.log(f'train/{tag}', loss_value, on_step=False, on_epoch=True)

        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        self.log_dict(self.__metric_manager.on_epoch_end('train'))

        

    def validation_epoch_end(self, outputs: Tuple[torch.tensor, dict]) -> None:
        """It's calling at the end of the validation epoch with the outputs of all validation steps."""
        total_loss, tagged_loss_values = outputs
        avg_loss = total_loss.mean()
        self.log('valid/avg_loss', avg_loss, on_step=False, on_epoch=True)

        for tag, loss_value in tagged_loss_values.items():
            self.log(f'valid/{tag}', loss_value, on_step=False, on_epoch=True)

        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        self.log_dict(self.__metric_manager.on_epoch_end('valid'))

<<<<<<< HEAD
    def test_epoch_end(self, outputs: torch.tensor) -> None:
        """Is calling at the end of a test epoch with the output of all test steps."""
        avg_loss = torch.stack(outputs).mean()
        self.log('test/loss', avg_loss, on_step=False, on_epoch=True)
        self.log_dict(self.__metric_manager.on_epoch_end('test'))
=======
    def test_epoch_end(self, outputs: Tuple[torch.tensor, dict]) -> None:
        """It's calling at the end of a test epoch with the output of all test steps."""
        total_loss, tagged_loss_values = outputs
        avg_loss = total_loss.mean()
        self.log('test/avg_loss', avg_loss, on_step=False, on_epoch=True)

        for tag, loss_value in tagged_loss_values.items():
            self.log(f'test/{tag}', loss_value, on_step=False, on_epoch=True)

        self.log_dict(self._metric_manager.on_epoch_end('test'))
>>>>>>> after review

    def to_onnx(self, onnx_params) -> None:
        """It's saving the model in ONNX format."""
        super().to_onnx(input_sample=self.example_input_array,
                        **onnx_params)

    def on_train_end(self) -> None:
<<<<<<< HEAD
        """Is calling at the end of training before logger experiment is closed."""
        onnx_params = self.__hparams.onnx_params
=======
        """It's calling at the end of training before logger experiment is closed."""
        onnx_params = self._hparams.onnx_params
>>>>>>> after review
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
        data_loader = self.__constructor.create_dataloaders('train')
        return data_loader

    def val_dataloader(self) -> Optional[List[torch.DataLoader]]:
        """Implement one or multiple PyTorch DataLoaders for prediction."""
        phase = 'valid'
<<<<<<< HEAD
        drop_last = self.__hparams['data'][phase][0]['dataloader']['drop_last']
        shuffle = self.__hparams['data'][phase][0]['dataloader']['shuffle']
        if shuffle or drop_last:
            raise ValueError(f'DataLoader parametrs `shuffle` and `drop_last` must be False in {phase} phase.')
=======
        drop_last = self._hparams['data'][phase][0]['dataloader']['drop_last']
        if drop_last:
            raise ValueError(f'DataLoader parametrs `drop_last` must be False in {phase} phase.')
>>>>>>> after review
        data_loader = self.__constructor.create_dataloaders(phase)
        return data_loader

    def test_dataloader(self) -> Optional[List[torch.DataLoader]]:
        """Implement one or multiple PyTorch DataLoaders for testing."""
        phase = 'test'
<<<<<<< HEAD
        drop_last = self.__hparams['data'][phase][0]['dataloader']['drop_last']
        shuffle = self.__hparams['data'][phase][0]['dataloader']['shuffle']
        if shuffle or drop_last:
            raise ValueError(f'DataLoader parametrs `shuffle` and `drop_last` must be False in {phase} phase.')
=======
        drop_last = self._hparams['data'][phase][0]['dataloader']['drop_last']
        if drop_last:
            raise ValueError(f'DataLoader parametrs `drop_last` must be False in {phase} phase.')
>>>>>>> after review
        data_loader = self.__constructor.create_dataloaders(phase)
        return data_loader

    def predict_dataloader(self) -> Optional[List[torch.DataLoader]]:
        """Implement one or multiple PyTorch DataLoaders for prediction."""
        phase = 'predict'
<<<<<<< HEAD
        drop_last = self.__hparams['data'][phase][0]['dataloader']['drop_last']
        shuffle = self.__hparams['data'][phase][0]['dataloader']['shuffle']
        if shuffle or drop_last:
            raise ValueError(f'DataLoader parametrs `shuffle` and `drop_last` must be False in {phase} phase.')
=======
        drop_last = self._hparams['data'][phase][0]['dataloader']['drop_last']
        if drop_last:
            raise ValueError(f'DataLoader parametrs `drop_last` must be False in {phase} phase.')
>>>>>>> after review
        data_loader = self.__constructor.create_dataloaders(phase)
        return data_loader

    def training_step_end(self, step_outputs: torch.tensor) -> torch.tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    def validation_step_end(self, step_outputs: torch.tensor) -> torch.tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    def test_step_end(self, step_outputs: torch.tensor) -> torch.tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    @property
    def hparams(self):
        """Is hyperparameters that set in yaml file."""
        return self.__hparams

    @property
    def metric_manager(self):
        """Is metric manager."""
        return self._metric_manager

    @property
    def criterion(self):
        """Is criterion."""
        return self._criterion

    @property
    def input_shapes(self):
        """Is input shape."""
        return self.__input_shapes
