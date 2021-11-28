import torch
from pydantic import BaseModel

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, DETECTION_NECKS, \
    DETECTION_HEADS, DETECTOR_INFER_MODULES
from .base_task import BaseTask

class DetectionTaskParams(BaseModel):
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    freeze_backbone: bool = False
    backbone_name: str
    backbone_params: dict = {}
    neck_name: str
    neck_params: dict = {}
    head_name: str
    head_params: dict = {}
    infer_name: str
    infer_params: dict = {}


@TASKS.register_class
class DetectionTask(BaseTask):
    config_parser = DetectionTaskParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)
        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params)

        neck_class = DETECTION_NECKS.get(self.params.neck_name)
        self.neck = neck_class(
                            in_channels=[128, 256, 512],
                            out_channels=128,
                            num_csp_blocks=1
        )
        
        head_class = DETECTION_HEADS.get(self.params.head_name)
        self.head = head_class(
                            num_classes=2, 
                            in_channels=128, 
                            feat_channels=128
        )

        infer_class = DETECTOR_INFER_MODULES.get(self.params.infer_name)
        self.infer_module = infer_class()


    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        output = self.head(neck_features)
        return output

    def forward_with_gt(self, batch):
        input_data = batch['input']

        gt_bboxes = batch['gt_bboxes']
        gt_labels = batch['gt_labels']


        heaed_prediction = self.forward(input_data)
        um_total_samples, \
            bbox_pred, bbox_targets,\
                 obj_pred, obj_targets,\
                      cls_pred, cls_targets = self.infer_module.forward_train(
                                                        heaed_prediction, 
                                                        gt_bboxe =gt_bboxes,
                                                        gt_labels=gt_labels
                                                        )
        output = {
            'bbox_pred': bbox_pred, 
            'bbox_target': bbox_targets, 
            'obj_pred': obj_pred, 
            'obj_target': obj_targets,
            'cls_pred': cls_pred, 
            'cls_targets': cls_targets            
            }
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        self.metric_manager.update('train', **output)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     output = self.forward_with_gt(batch)
    #     loss = self.criterion(**output)
    #     self.metric_manager.update('valid', **output)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     output = self.forward_with_gt(batch)
    #     loss = self.criterion(**output)
    #     self.metric_manager.update('test', **output)
    #     return loss

    def configure_optimizers(self):
        modules = [self.neck, self.head]
        if not self.params.freeze_backbone:
            modules.append(self.backbone)
        optimizer = create_optimizer(modules, self.hparams.optimizers)
        if self.hparams.schedulers is not None:
            scheduler = create_scheduler(optimizer, self.hparams.schedulers)
            return [optimizer], [scheduler]
        else:
            return [optimizer]
