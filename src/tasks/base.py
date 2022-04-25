from functools import partial
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from src.constructor import create_scheduler, create_optimizer, create_dataset, JointLoss
from src.constructor.config_structure import TrainConfigParams
from src.metrics import MetricManager
from src.models.backbones.utils import load_checkpoint


@dataclass
class BaseTaskParams:
    input_shapes: dict
    checkpoint: str = None
    onnx_params: dict = None
    inference_mode: bool = False


class BaseTask(LightningModule, ABC):
    """An abstract class that represent main methods of tasks"""

    def __init__(self, hparams: TrainConfigParams):
        """
        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.__params = self.config_parser(**hparams.task.params)
        self.__metric_manager = MetricManager(self.hparams.metrics)
        self._criterion = JointLoss(self, self.hparams.loss)
        self.__input_shapes = self.params.input_shapes
        self.__inference_mode = self.params.inference_mode
        self.example_input_array = [torch.rand(1, *shape) for _, shape in self.__input_shapes.items()]

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward_with_gt(batch: dict) -> dict:
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
        if self.params.checkpoint is not None:
            load_checkpoint(self, self.params.checkpoint, strict=False)

    def on_test_start(self) -> None:
        if self.params.checkpoint is not None:
            load_checkpoint(self, self.params.checkpoint, strict=False)

    def training_epoch_end(self, outputs: torch.tensor) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train/loss', avg_loss, on_step=False, on_epoch=True)
        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        self.log_dict(self.__metric_manager.on_epoch_end('train'))

    def validation_epoch_end(self, outputs: torch.tensor) -> None:
        avg_loss = torch.stack(outputs).mean()
        self.log('valid/loss', avg_loss, on_step=False, on_epoch=True)
        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        self.log_dict(self.__metric_manager.on_epoch_end('valid'))

    def test_epoch_end(self, outputs: torch.tensor) -> None:
        avg_loss = torch.stack(outputs).mean()
        self.log('test/loss', avg_loss, on_step=False, on_epoch=True)
        self.log_dict(self.__metric_manager.on_epoch_end('test'))

    def to_onnx(self) -> None:
        super().to_onnx(input_sample=self.example_input_array,
                        **self.params.onnx_params)

    def on_train_end(self) -> None:
        if self.params.onnx_params is not None:
            self.to_onnx()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.parameters(), self.hparams.optimizers)
        if self.hparams.schedulers is not None:
            scheduler = create_scheduler(optimizer, self.hparams.schedulers)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    @staticmethod
    def prepare_dataloader(dataset_params, common_dataset_params, dataloader_params) -> torch.DataLoader:
        dataset = create_dataset(dataset_params.name, common_dataset_params, dataset_params)

        use_custom_collate_fn = dataloader_params.use_custom_collate_fn and hasattr(dataset, 'collate_fn')
        collate_fn = dataset.collate_fn if use_custom_collate_fn else None
        if dataloader_params.use_custom_batch_sampler:
            batch_sampler = dataset.batch_sampler(batch_size=dataloader_params.batch_size,
                                                  shuffle=dataloader_params.shuffle,
                                                  drop_last=dataloader_params.drop_last)
            Loader = partial(DataLoader, batch_sampler=batch_sampler)
        else:
            Loader = partial(DataLoader, batch_size=dataloader_params.batch_size,
                             shuffle=dataloader_params.shuffle, drop_last=dataloader_params.drop_last,
                             pin_memory=dataloader_params.pin_memory)
        loader = Loader(dataset=dataset,
                        num_workers=dataloader_params.num_workers,
                        collate_fn=collate_fn)
        return loader

    def train_dataloader(self) -> torch.DataLoader:
        data_params = self.hparams.data

        if data_params.train_params is None:
            return None

        data_loader = self.prepare_dataloader(data_params.train_params, data_params.common_params,
                                              data_params.train_params.dataloader_params)

        return data_loader

    def val_dataloader(self) -> torch.DataLoader:
        data_params = self.hparams.data

        if data_params.valid_params is None:
            return None

        dataloader_params = data_params.valid_params.dataloader_params.copy()
        dataloader_params.shuffle = False
        dataloader_params.drop_last = False

        data_loader = self.prepare_dataloader(data_params.valid_params, data_params.common_params,
                                              dataloader_params)

        return data_loader

    def test_dataloader(self) -> torch.DataLoader:
        data_params = self.hparams.data

        if data_params.test_params is None:
            return None

        dataloader_params = data_params.test_params.dataloader_params.copy()
        dataloader_params.shuffle = False
        dataloader_params.drop_last = False

        data_loader = self.prepare_dataloader(data_params.test_params, data_params.common_params,
                                              dataloader_params)

        return data_loader

    def training_step_end(self, step_outputs: torch.tensor) -> torch.tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    def validation_step_end(self, step_outputs: torch.tensor) -> torch.tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    def test_step_end(self, step_outputs: torch.tensor) -> torch.tensor:
        return step_outputs.mean(dim=0, keepdim=True)

    @property
    def params(self):
        return self.__params

    @property
    def  metric_manager(self):
        return self.__metric_manager

    @property
    def criterion(self):
        return self._criterion

    @property
    def input_shapes(self):
        return self.__input_shapes

    @property
    def inference_mode(self):
        return self.__inference_mode
