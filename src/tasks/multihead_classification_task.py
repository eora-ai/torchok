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
    target: str
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
class MultiHeadClassificationTask(BaseTask):
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
        self.target_mapping = {}
        for head in heads:
            head_type = CLASSIFICATION_HEADS.get(head.type)
            head_name = head.name
            target = head.target
            head_params = head.params

            head_params['in_features'] = self.pooling.out_features
            self.heads[head_name] = head_type(**head_params)
            self.target_mapping[head_name] = target

    def forward(self, x):
        features = self.backbone(x)
        features = self.pooling(features)
        return features

    def forward_with_gt(self, batch):
        features = self.backbone(batch['input'])
        features = self.pooling(features)
        output = {'embeddings': features}
        for head_name, head in self.heads.items():
            target_name = f'target_{self.target_mapping[head_name]}'
            head_target = batch[target_name]

            out = head(features, head_target)
            output[f'prediction_{head_name}'] = out
            output[target_name] = head_target

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
