from typing import Any, Dict, Tuple, List, Optional, Union
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from omegaconf import DictConfig

from torchok.constructor.config_structure import Phase
from torchok.constructor.constructor import Constructor
from torchok.constructor.load import load_checkpoint


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
        self._input_tensors = []
        self._losses = self.__constructor.configure_losses() if hparams.get('joint_loss') is not None else None
        self._hparams = hparams
        self._metrics_manager = self.__constructor.configure_metrics_manager()
        
        # `inputs` key in yaml used for model checkpointing.
        inputs = hparams.task.params.get('inputs')
        if inputs is not None:
            for input_params in inputs:
                input_tensor = torch.rand(*input_params['shape']).type(torch.__dict__[input_params['dtype']])
                self._input_tensors.append(input_tensor)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Abstract forward method for validation an test."""
        pass

    @abstractmethod
    def forward_with_gt(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Abstract forward method for training(with ground truth labels)."""
        pass

    def train_modules(self) -> List[str]:
        """Create a list of 1st-level modules that need to be optimized. The method is used to apply an optimizer
        for the returned modules.

        By default, it would be self.children().

        Returns: List of modules that need to be optimized.
        """
        return self.children()

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configure optimizers."""
        modules = self.train_modules()
        opt_sched_list = self.__constructor.configure_optimizers(modules)
        return opt_sched_list

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

        self.__check_drop_last_params(data_params, Phase.VALID.value)

        data_loader = self.__constructor.create_dataloaders(Phase.VALID)
        return data_loader

    def test_dataloader(self) -> Optional[List[DataLoader]]:
        """Implement one or multiple PyTorch DataLoaders for testing."""
        data_params = self._hparams['data'].get(Phase.TEST, None)

        if data_params is None:
            return None

        self.__check_drop_last_params(data_params, Phase.TEST.value)

        data_loader = self.__constructor.create_dataloaders(Phase.TEST)
        return data_loader

    def predict_dataloader(self) -> Optional[List[DataLoader]]:
        """Implement one or multiple PyTorch DataLoaders for prediction."""
        data_params = self._hparams['data'].get(Phase.PREDICT, None)

        if data_params is None:
            return None

        self.__check_drop_last_params(data_params, Phase.PREDICT.value)

        data_loader = self.__constructor.create_dataloaders(Phase.PREDICT)
        return data_loader

    def __check_drop_last_params(self, data_params: List[Dict[str, Any]], phase: str) -> None:
        for data_param in data_params:
            drop_last = data_param['dataloader'].get('drop_last', False)
            if drop_last:
                # TODO: create logger and print a warning instead
                raise ValueError(f'DataLoader parameters `drop_last` must be False in {phase} phase.')

    def on_train_start(self) -> None:
        if self.current_epoch == 0:
            load_checkpoint(self, base_ckpt_path=self._hparams.task.base_checkpoint, 
                            overridden_name2ckpt_path=self._hparams.task.overridden_checkpoints,
                            exclude_keys=self._hparams.task.exclude_keys)

    def on_test_start(self) -> None:
        load_checkpoint(self, base_ckpt_path=self._hparams.task.base_checkpoint, 
                        overridden_name2ckpt_path=self._hparams.task.overridden_checkpoints,
                        exclude_keys=self._hparams.task.exclude_keys)

    def training_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete training loop."""
        output = self.forward_with_gt(batch[0])
        total_loss, tagged_loss_values = self._losses(**output)
        self._metrics_manager.forward(Phase.TRAIN, **output)
        output_dict = {'loss': total_loss}
        output_dict.update(tagged_loss_values)
        return output_dict

    def validation_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Complete validation loop."""
        output = self.forward_with_gt(batch)
        self._metrics_manager.forward(Phase.VALID, **output)

        # In arcface classification task, if we try to compute loss on test dataset with different number
        # of classes we will crash the train study.
        if self._hparams.task.compute_loss_on_valid:
            total_loss, tagged_loss_values = self._losses(**output)
            output_dict = {'loss': total_loss}
            output_dict.update(tagged_loss_values)
        else:
            output_dict = {}
        
        return output_dict

    def test_step(self, batch: Dict[str, Union[torch.Tensor, int]], batch_idx: int) -> None:
        """Complete test loop."""
        output = self.forward_with_gt(batch)
        self._metrics_manager.forward(Phase.TEST, **output)

    def training_step_end(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = {tag: value.mean() for tag, value in self.all_gather(outputs, sync_grads=True).items()}

        for tag, value in output_dict.items():
            self.log(f'train/{tag}', value, on_step=True, on_epoch=False)

        return output_dict

    def validation_step_end(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = {tag: value.mean() for tag, value in self.all_gather(outputs).items()}

        for tag, value in output_dict.items():
            self.log(f'valid/{tag}', value, on_step=True, on_epoch=False)

        return output_dict

    def training_epoch_end(self,
                           training_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """It's calling at the end of the training epoch with the outputs of all training steps."""
        self.log_dict(self._metrics_manager.on_epoch_end(Phase.TRAIN))

    def validation_epoch_end(self,
                             valid_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """It's calling at the end of the validation epoch with the outputs of all validation steps."""
        self.log_dict(self._metrics_manager.on_epoch_end(Phase.VALID))

    def test_epoch_end(self,
                       test_step_outputs: List[Dict[str, torch.Tensor]]) -> None:
        """It's calling at the end of a test epoch with the output of all test steps."""
        self.log_dict(self._metrics_manager.on_epoch_end(Phase.TEST))

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
    def input_shapes(self) -> List[List[int]]:
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
