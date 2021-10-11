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

        self.dataset_train, self.dataset_valid, self.dataset_test = None, None, None

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

    def setup(self, stage: str = None):
        data_params = self.hparams.data
        common_params = data_params.common_params

        train_params = data_params.train_params
        valid_params = data_params.valid_params
        test_params = data_params.test_params

        self.dataset_train = create_dataset(train_params.name, common_params, train_params)
        self.dataset_valid = create_dataset(valid_params.name, common_params, valid_params)

        if test_params is None:
            self.dataset_test = None
        else:
            self.dataset_test = create_dataset(test_params.name, common_params, test_params)

    @staticmethod
    def prepare_dataloader(dataset, params):
        use_custom_collate_fn = params.use_custom_collate_fn and hasattr(dataset, 'collate_fn')
        collate_fn = dataset.collate_fn if use_custom_collate_fn else None
        if params.use_custom_batch_sampler:
            batch_sampler = dataset.batch_sampler(batch_size=params.batch_size,
                                                  shuffle=params.shuffle,
                                                  drop_last=params.drop_last)
            Loader = partial(DataLoader, batch_sampler=batch_sampler)
        else:
            Loader = partial(DataLoader, batch_size=params.batch_size,
                             shuffle=params.shuffle, drop_last=params.drop_last)
        loader = Loader(dataset=dataset,
                        num_workers=params.num_workers,
                        collate_fn=collate_fn)
        return loader

    def train_dataloader(self):
        return self.prepare_dataloader(self.dataset_train, self.hparams.data.train_params.dataloader_params)

    def val_dataloader(self):
        dataloader_params = self.hparams.data.valid_params.dataloader_params
        dataloader_params.shuffle = False
        dataloader_params.drop_last = False
        return self.prepare_dataloader(self.dataset_valid, dataloader_params)

    def test_dataloader(self):
        dataset = self.dataset_test
        if dataset:
            dataloader_params = self.hparams.data.test_params.dataloader_params
            dataloader_params.shuffle = False
            dataloader_params.drop_last = False
            return self.prepare_dataloader(dataset, dataloader_params)

    def training_step_end(self, batch_parts_outputs):
        return batch_parts_outputs.mean(dim=0, keepdim=True)

    def validation_step_end(self, batch_parts_outputs):
        return batch_parts_outputs.mean(dim=0, keepdim=True)

    def test_step_end(self, batch_parts_outputs):
        return batch_parts_outputs.mean(dim=0, keepdim=True)
