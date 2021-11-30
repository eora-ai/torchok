from typing import DefaultDict
import torch
from pydantic import BaseModel

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, DETECTION_NECKS, \
    DETECTION_HEADS, DETECTOR_INFER_MODULES
from .base_task import BaseTask
import torch.nn as nn

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
class DetectionTask(BaseTask, nn.Module):
    config_parser = DetectionTaskParams

    def __init__(self, hparams: TrainConfigParams):
        super(DetectionTask, self).__init__(hparams)
        # super().__init__(hparams)
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
        self.infer_module = infer_class(num_classes=2)


    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        cls_score, bbox_pred, objectness = self.head(neck_features)
        return cls_score, bbox_pred, objectness

    def forward_with_gt(self, batch):
        input_data = batch['input']

        gt_bboxes = batch['target_bboxes']
        gt_labels = batch['target_labels']

        # print('gt_bboxes = ' + str(gt_bboxes.shape))
        # print(gt_bboxes)
        # print('gt_labels = ' + str(gt_labels.shape))
        # print('intput = ' + str(input_data.shape))

        cls_score, bbox_pred, objectness = self.forward(input_data)
        um_total_samples, \
            bbox_pred, bbox_targets,\
                 obj_pred, obj_targets,\
                      cls_pred, cls_targets = self.infer_module.forward_train(
                                                        cls_score,
                                                        bbox_pred,
                                                        objectness, 
                                                        gt_bboxes=gt_bboxes,
                                                        gt_labels=gt_labels
                                                        )
        
        # print('obj_pred = ' + str(obj_pred))
        # print(type(obj_pred))
        # print(obj_pred.shape)
        # print('obj_targets = ' + str(obj_targets))
        # print(type(obj_targets))
        # print(obj_targets.shape)
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

    def validation_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        self.metric_manager.update('valid', **output)
        return loss

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
