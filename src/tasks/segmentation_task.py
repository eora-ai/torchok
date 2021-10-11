import torch
from pydantic import BaseModel

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, SEGMENTATION_HEADS, SEGMENTATION_MODELS
from .base_task import BaseTask


class AbstractSegmentationTask(BaseTask):

    def forward(self, x):
        return NotImplementedError()

    def forward_with_gt(self, batch):
        input_data = batch['input']
        segm_logits = self.forward(input_data)
        if isinstance(segm_logits, (list, tuple)):
            output = [{'target': batch['target'], 'prediction': out} for out in segm_logits]
        else:
            output = {'target': batch['target'], 'prediction': segm_logits}
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        if isinstance(output, (list, tuple)):
            loss = 0
            for out in output:
                loss += self.criterion(**out)
                self.metric_manager.update('train', **out)
        else:
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


class StandaloneSegmentationParams(BaseModel):
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    segmenter_name: str
    segmenter_params: dict = {}
    freeze_backbone: bool = False


@TASKS.register_class
class StandaloneSegmentationTask(AbstractSegmentationTask):
    config_parser = StandaloneSegmentationParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)
        self.segmenter = SEGMENTATION_MODELS.get(self.params.segmenter_name)(**self.params.segmenter_params)

    def forward(self, x):
        return self.segmenter(x)


class SegmentationTaskParams(StandaloneSegmentationParams):
    backbone_name: str
    backbone_params: dict = {}


@TASKS.register_class
class SegmentationTask(AbstractSegmentationTask):
    config_parser = SegmentationTaskParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)
        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params)
        segmenter_class = SEGMENTATION_HEADS.get(self.params.segmenter_name)
        self.head = segmenter_class(encoder_channels=self.backbone.encoder_channels,
                                    **self.params.segmenter_params)

    def configure_optimizers(self):
        modules = [self.head]
        if not self.params.freeze_backbone:
            modules.append(self.backbone)
        optimizer = create_optimizer(modules, self.hparams.optimizers)
        if self.hparams.schedulers is not None:
            scheduler = create_scheduler(optimizer, self.hparams.schedulers)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def forward(self, x):
        with torch.set_grad_enabled(not self.params.freeze_backbone and self.training):
            last_features, backbone_features = self.backbone.forward_backbone_features(x)
        segm_logits = self.head(backbone_features)

        return segm_logits
