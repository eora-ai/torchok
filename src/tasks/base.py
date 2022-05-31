from typing import Any, Dict, Tuple, List, Optional, Union
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from omegaconf import DictConfig

from src.constructor.config_structure import Phase
from src.constructor.constructor import Constructor


class BaseTask(LightningModule, ABC):
    """An abstract class that represent main methods of tasks."""

    def __init__(self, hparams: DictConfig):
        """Init BaseTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self._input_tensors = []
        self._losses = self.__constructor.configure_losses()
        self._hparams = hparams
        self._metrics_manager = self.__constructor.configure_metrics_manager()
        self.__constructor = Constructor(hparams)
        self.__input_shapes = self._hparams.task.params.input_shapes
        self.__input_dtypes = self._hparams.task.params.input_dtypes

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

    def training_epoch_end(self,
                           training_step_outputs: List[Dict[str, Union[torch.Tensor, Dict[str, Dict]]]]) -> None:
        """It's calling at the end of the training epoch with the outputs of all training steps."""
        self.log_dict(self._metrics_manager.on_epoch_end(Phase.TRAIN))

    def validation_epoch_end(self,
                             valid_step_outputs: List[Dict[str, Union[torch.Tensor, Dict[str, Dict]]]]) -> None:
        """It's calling at the end of the validation epoch with the outputs of all validation steps."""
        total_loss = torch.stack([x['loss'] for x in valid_step_outputs]).mean()
        self.log('valid/total_loss', total_loss, on_step=False, on_epoch=True)

        for tag in valid_step_outputs[0].keys():
            if tag == 'loss':
                continue

            loss = torch.stack([x[tag] for x in valid_step_outputs]).mean()
            self.log(f'valid/{tag}', loss, on_step=False, on_epoch=True)

        self.log_dict(self._metrics_manager.on_epoch_end(Phase.VALID))

    def test_epoch_end(self,
                       test_step_outputs: List[Dict[str, Union[torch.Tensor, Dict[str, Dict]]]]) -> None:
        """It's calling at the end of a test epoch with the output of all test steps."""
        self.log_dict(self._metrics_manager.on_epoch_end(Phase.TEST))

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configure optimizers."""
        configure_optimizers = self.__constructor.configure_optimizers(self.parameters())
        return configure_optimizers

    def train_dataloader(self) -> Optional[List[DataLoader]]:
        """Implement one or more PyTorch DataLoaders for training."""
        data_params = self._hparams['data'].get(Phase.TRAIN, None)

        if data_params is None:
            return None

        data_loader = self.__constructor.create_dataloaders(Phase.TRAIN)
        return data_loader

    def val_dataloader(self) -> Optional[List[DataLoader]]:
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

    def test_dataloader(self) -> Optional[List[DataLoader]]:
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

    def predict_dataloader(self) -> Optional[List[DataLoader]]:
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

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> dict:
        """Complete training loop."""
        output = self.forward_with_gt(batch[0])
        loss = self._losses(**output)
        self._metrics_manager.forward(Phase.TRAIN, **output)
        output_dict = {'loss': loss[0]}
        output_dict.update(loss[1])

        return output_dict

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> dict:
        """Complete validation loop."""
        output = self.forward_with_gt(batch)
        loss = self._losses(**output)
        self._metrics_manager.forward(Phase.VALID, **output)
        output_dict = {'loss': loss[0]}
        output_dict.update(loss[1])

        return output_dict

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx) -> None:
        """Complete test loop."""
        output = self.forward_with_gt(batch)
        self._metrics_manager.forward(Phase.TEST, **output)

    def training_step_end(self, outputs):
        output_dict = {tag: value.mean() for tag, value in self.all_gather(outputs, sync_grads=True).items()}

        for tag, value in output_dict.items():
            if tag == 'loss':
                self.log('train/total_loss', value, on_step=True, on_epoch=False)
            else:
                self.log(f'train/{tag}', value, on_step=True, on_epoch=False)

        return output_dict

    def validation_step_end(self, outputs):
        output_dict = {tag: value.mean() for tag, value in self.all_gather(outputs).items()}

        return output_dict

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
    def input_shapes(self) -> List[Tuple[int, int, int, int]]:
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
