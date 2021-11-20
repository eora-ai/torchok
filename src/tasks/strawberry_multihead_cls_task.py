from typing import List

import torch.nn as nn
from pydantic import BaseModel

from src.constructor.config_structure import TrainConfigParams
from src.constructor import create_backbone
from src.registry import TASKS, CLASSIFICATION_HEADS, POOLINGS
from .base_task import BaseTask


class HeadParams(BaseModel):
    type: str
    name: str
    params: dict = {}


class MultiHeadClassificationParams(BaseModel):
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    backbone_name: str
    backbone_params: dict = {}
    pooling_name: str = 'PoolingLinear'
    # in_features is taken from backbone.num_features
    # Be careful, default pooling parameters dict is not full!
    pooling_params: dict = {}
    heads: List[HeadParams]


@TASKS.register_class
class StrawberryMultiHeadCLSTask(BaseTask):
    config_parser = MultiHeadClassificationParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)
        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params)

        self.params.pooling_params['in_features'] = self.backbone.num_features
        self.pooling = POOLINGS.get(self.params.pooling_name)(**self.params.pooling_params)

        heads = self.params.heads
        self.heads = nn.ModuleDict()
        self.head_tasks = {}
        for head in heads:
            head_type = CLASSIFICATION_HEADS.get(head.type)
            head_name = head.name
            head_params = head.params

            head_params['in_features'] = self.pooling.out_features
            self.heads[head_name] = head_type(**head_params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        features = self.pooling(features)
        classification_output = self.heads['classification'](features)
        regression_output = self.heads['regression'](features)
        regression_output = self.sigmoid(regression_output)
        return classification_output, regression_output

    def forward_with_gt(self, batch):
        cl_out, reg_out = self.forward(batch['input'])
        output = {
            'cls_target': batch['cls_target'], 
            'reg_target': batch['reg_target'],
            'cls_prediction': cl_out,
            'reg_prediction': reg_out
            }
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
        # OPTIONAL
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        self.metric_manager.update('test', **output)
        return loss
