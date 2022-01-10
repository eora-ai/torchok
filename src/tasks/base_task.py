from functools import partial

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from src.constructor import create_scheduler, create_optimizer, create_dataset, JointLoss
from src.constructor.config_structure import TrainConfigParams
from src.metrics import MetricManager
from src.models.backbones.utils import load_checkpoint


class BaseTask(LightningModule):
    def __init__(self, hparams: TrainConfigParams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.params = self.config_parser(**hparams.task.params)

        self.metric_manager = MetricManager(self.hparams.metrics)
        self.criterion = JointLoss(self, self.hparams.loss)
        self.example_input_array = torch.rand(1, *self.params.input_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def on_train_start(self) -> None:
        if self.params.checkpoint is not None:
            load_checkpoint(self, self.params.checkpoint, strict=False)

    def on_test_start(self) -> None:
        if self.params.checkpoint is not None:
            load_checkpoint(self, self.params.checkpoint, strict=False)

    def training_step(self, *args, **kwargs):
        raise NotImplementedError()

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train/loss', avg_loss, on_step=False, on_epoch=True)
        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        self.log_dict(self.metric_manager.on_epoch_end('train'))

    def validation_step(self, *args, **kwargs):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('valid/loss', avg_loss, on_step=False, on_epoch=True)
        self.log('step', self.current_epoch, on_step=False, on_epoch=True)
        self.log_dict(self.metric_manager.on_epoch_end('valid'))

    def test_step(self, *args, **kwargs):
        raise NotImplementedError()

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('test/loss', avg_loss, on_step=False, on_epoch=True)
        self.log_dict(self.metric_manager.on_epoch_end('test'))

    def configure_optimizers(self):
        optimizer = create_optimizer(self.parameters(), self.hparams.optimizers)
        if self.hparams.schedulers is not None:
            scheduler = create_scheduler(optimizer, self.hparams.schedulers)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    @staticmethod
    def prepare_dataloader(dataset_params, common_dataset_params, dataloader_params):
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
                             shuffle=dataloader_params.shuffle, drop_last=dataloader_params.drop_last)
        loader = Loader(dataset=dataset,
                        num_workers=dataloader_params.num_workers,
                        collate_fn=collate_fn)
        return loader

    def train_dataloader(self):
        data_params = self.hparams.data

        if data_params.train_params is None:
            return None

        data_loader = self.prepare_dataloader(data_params.train_params, data_params.common_params,
                                              data_params.train_params.dataloader_params)

        return data_loader

    def val_dataloader(self):
        data_params = self.hparams.data

        if data_params.valid_params is None:
            return None

        dataloader_params = data_params.valid_params.dataloader_params.copy()
        dataloader_params.shuffle = False
        dataloader_params.drop_last = False

        data_loader = self.prepare_dataloader(data_params.valid_params, data_params.common_params,
                                              dataloader_params)

        return data_loader

    def test_dataloader(self):
        data_params = self.hparams.data

        if data_params.test_params is None:
            return None

        dataloader_params = data_params.test_params.dataloader_params.copy()
        dataloader_params.shuffle = False
        dataloader_params.drop_last = False

        data_loader = self.prepare_dataloader(data_params.test_params, data_params.common_params,
                                              dataloader_params)

        return data_loader

    def training_step_end(self, batch_parts_outputs):
        return batch_parts_outputs.mean(dim=0, keepdim=True)

    def validation_step_end(self, batch_parts_outputs):
        return batch_parts_outputs.mean(dim=0, keepdim=True)

    def test_step_end(self, batch_parts_outputs):
        return batch_parts_outputs.mean(dim=0, keepdim=True)
