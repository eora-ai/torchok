from pydantic import BaseModel
from typing import Optional

import torch

from src.constructor.config_structure import TrainConfigParams
from src.constructor import create_backbone
from src.registry import TASKS, HEADS, POOLINGS
from .base_task import BaseTask


class SimCLR_Params(BaseModel):
    checkpoint: str = None
    input_size: list  # [numchans, height, width]
    backbone_name: str
    backbone_params: dict = {}
    pooling_name: Optional[str] = None
    # in_features is taken from backbone.num_features
    # Be careful, default pooling parameters dict is not full!
    pooling_params: dict = {}
    head_name: str
    head_params: dict


@TASKS.register_class
class SimCLR(BaseTask):
    """
    Task-agnostic part of the SimCLR v2 approach described in paper
    `Big Self-Supervised Models are Strong Semi-Supervised Learners`_

    .. _Big Self-Supervised Models are Strong Semi-Supervised Learners:
            https://arxiv.org/abs/2006.10029
    """
    config_parser = SimCLR_Params

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)
        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params)

        if self.params.pooling_name is not None:
            self.params.pooling_params['in_features'] = self.backbone.num_features
            self.pooling = POOLINGS.get(self.params.pooling_name)(**self.params.pooling_params)
        else:
            self.pooling = None

        self.params.head_params['in_features'] = self.backbone.num_features
        self.head = HEADS.get(self.params.head_name)(**self.params.head_params)

    def forward(self, x):
        representation = self.backbone(x)
        if self.pooling:
            representation = self.pooling(representation)
        emb = self.head(representation)

        return emb

    def forward_with_gt(self, batch):
        x1, x2 = batch['input_0'], batch['input_1']
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)

        return emb1, emb2

    def training_step(self, batch, batch_idx):
        emb1, emb2 = self.forward_with_gt(batch)
        loss = self.calc_loss(emb1=emb1, emb2=emb2)

        return loss

    def calc_loss(self, emb1, emb2):
        loss = self.criterion(emb1=emb1, emb2=emb2)

        return loss

    def validation_step(self, batch, batch_idx):
        emb1, emb2 = self.forward_with_gt(batch)
        loss = self.calc_loss(emb1=emb1, emb2=emb2)

        return loss

    def test_step(self, batch, batch_idx):
        emb1, emb2 = self.forward_with_gt(batch)
        loss = self.calc_loss(emb1=emb1, emb2=emb2)

        return loss
