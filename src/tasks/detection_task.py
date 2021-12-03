from typing import DefaultDict
import torch
from pydantic import BaseModel

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, DETECTION_NECKS, \
    DETECTION_HEADS, DETECTION_HAT
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
    hat_name: str
    hat_params: dict = {}


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
                            num_csp_blocks=2
        )
        
        head_class = DETECTION_HEADS.get(self.params.head_name)
        self.head = head_class(
                            num_classes=self.params.head_params['num_classes'], 
                            in_channels=128, 
                            feat_channels=128
        )

        infer_class = DETECTION_HAT.get(self.params.hat_name)
        self.infer_module = infer_class(num_classes=self.params.head_params['num_classes'])
        self.num_classes = self.params.head_params['num_classes']

    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        cls_score, bbox_pred, objectness = self.head(neck_features)
        return cls_score, bbox_pred, objectness

    def forward_with_gt(self, batch):
        input_data = batch['input']
        gt_bboxes = batch['target_bboxes']
        gt_labels = batch['target_labels']

        cls_score, bbox_pred, objectness = self.forward(input_data)

        num_total_samples, \
            loss_bbox_pred, loss_bbox_targets,\
                loss_obj_pred, loss_obj_targets,\
                    loss_cls_pred, loss_cls_targets = self.infer_module.forward_train(
                                                            cls_score,
                                                            bbox_pred,
                                                            objectness, 
                                                            gt_bboxes=gt_bboxes,
                                                            gt_labels=gt_labels
                                                            )
        
        output = {
            'bbox_pred': loss_bbox_pred, 
            'bbox_target': loss_bbox_targets, 
            'obj_pred': loss_obj_pred, 
            'obj_target': loss_obj_targets,
            'cls_pred': loss_cls_pred, 
            'cls_targets': loss_cls_targets,
            }
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)

        input_data = batch['input']
        gt_bboxes = batch['target_bboxes']
        gt_labels = batch['target_labels']
        cls_score, bbox_pred, objectness = self.forward(input_data)
        prediction = self.infer_module.forward_infer(cls_score, bbox_pred, objectness)
        target = [[gt_bboxes[i], gt_labels[i]] for i in range(gt_bboxes.shape[0])]
        valid_output = {
            'target': target,
            'prediction': prediction
        }
        self.metric_manager.update('train', **valid_output)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)

        input_data = batch['input']
        gt_bboxes = batch['target_bboxes']
        gt_labels = batch['target_labels']
        cls_score, bbox_pred, objectness = self.forward(input_data)
        prediction = self.infer_module.forward_infer(cls_score, bbox_pred, objectness)
        target = [[gt_bboxes[i], gt_labels[i]] for i in range(gt_bboxes.shape[0])]
        valid_output = {
            'target': target,
            'prediction': prediction
        }
        self.metric_manager.update('valid', **valid_output)
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
