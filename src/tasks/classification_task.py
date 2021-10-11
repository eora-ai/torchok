import torch
from pydantic import BaseModel

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, CLASSIFICATION_HEADS, POOLINGS
from .base_task import BaseTask


class ClassificationParams(BaseModel):
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    backbone_name: str
    backbone_params: dict = {}
    pooling_name: str = 'PoolingLinear'
    # in_features is taken from backbone.num_features
    # Be careful, default pooling parameters dict is not full!
    pooling_params: dict = {}
    head_name: str
    head_params: dict
    freeze_backbone: bool = False


@TASKS.register_class
class ClassificationTask(BaseTask):
    config_parser = ClassificationParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)

        # create model
        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params)

        self.params.pooling_params['in_features'] = self.backbone.num_features
        self.pooling = POOLINGS.get(self.params.pooling_name)(**self.params.pooling_params)

        # create classification head
        self.params.head_params['in_features'] = self.pooling.out_features
        self.head = CLASSIFICATION_HEADS.get(self.params.head_name)(**self.params.head_params)

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def configure_optimizers(self):
        modules = [self.pooling, self.head]
        if not self.params.freeze_backbone:
            modules.append(self.backbone)
        optimizer = create_optimizer(modules, self.hparams.optimizers)
        if self.hparams.schedulers is not None:
            scheduler = create_scheduler(optimizer, self.hparams.schedulers)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def forward_with_gt(self, batch):
        input_data = batch['input']
        target = batch['target']
        with torch.set_grad_enabled(not self.params.freeze_backbone and self.training):
            features = self.backbone(input_data)
        features = self.pooling(features)
        prediction = self.head(features, target)
        output = {'target': target, 'embeddings': features, 'prediction': prediction}
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        self.metric_manager.update('train', **output)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        self.metric_manager.update('valid', **output)
        return loss

    def test_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        self.metric_manager.update('test', **output)
        return loss
