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
                            num_csp_blocks=2
        )
        
        head_class = DETECTION_HEADS.get(self.params.head_name)
        self.head = head_class(
                            num_classes=self.params.head_params['num_classes'], 
                            in_channels=128, 
                            feat_channels=128
        )

        infer_class = DETECTOR_INFER_MODULES.get(self.params.infer_name)
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

        # print('gt_bboxes = ' + str(gt_bboxes.shape))
        # print(gt_bboxes)
        # print('gt_labels = ' + str(gt_labels.shape))
        # print('intput = ' + str(input_data.shape))

        cls_score, bbox_pred, objectness = self.forward(input_data)
        # print('cls shape ' + str(cls_score[0].shape))
        # print('bbox_pred shape = ' + str(bbox_pred[0].shape))
        # print('objectness shape ' + str(objectness[0].shape))

        num_total_samples, \
            bbox_pred, bbox_targets,\
                obj_pred, obj_targets,\
                    cls_pred, cls_targets, ious, scores = self.infer_module.forward_train(
                                                            cls_score,
                                                            bbox_pred,
                                                            objectness, 
                                                            gt_bboxes=gt_bboxes,
                                                            gt_labels=gt_labels
                                                            )
        
        # print('bbox target = ' + str(bbox_targets))
        # print('bbox_pred = ' + str(bbox_pred))

        # print('target_labels = ' + str(cls_targets))
        # print('pred_labels = ' + str(cls_pred))

        print('ious = ' + str(ious))
        print('scores = ' + str(scores))

        metric_cls_targets = cls_targets.clone()
        metric_cls_pred = cls_pred.clone().detach()

        

        if metric_cls_pred.nelement() != 0 and metric_cls_targets.nelement() != 0:
            if self.num_classes != 1:
                metric_cls_targets = torch.argmax(metric_cls_targets, axis = -1)
                metric_cls_pred = torch.argmax(metric_cls_pred, axis = -1)
            else:
                metric_cls_targets = metric_cls_targets.squeeze(-1)
                metric_cls_targets = torch.ones_like(metric_cls_targets,  dtype=torch.int64)

                metric_cls_pred = metric_cls_pred.squeeze(-1)
                # print(' metric_cls_pred ' + str(metric_cls_pred))
                backgroudn_indexes = torch.where(metric_cls_pred < 0)[0]
                metric_cls_pred = torch.ones_like(metric_cls_pred,  dtype=torch.int64)
                metric_cls_pred[backgroudn_indexes] = 0

            scores = scores.squeeze(-1)
        else:
            metric_cls_targets = torch.tensor([])
            metric_cls_pred = torch.tensor([])
            scores = torch.tensor([])
        # print('metric cls target = ' + str(metric_cls_targets))
        # print('metric cls pred = ' + str(metric_cls_pred))
        # print('scores = ' + str(scores))

        output = {
            'bbox_pred': bbox_pred, 
            'bbox_target': bbox_targets, 
            'obj_pred': obj_pred, 
            'obj_target': obj_targets,
            'cls_pred': cls_pred, 
            'cls_targets': cls_targets,
            'num_total_samples': num_total_samples,    
            'metric_target': dict(boxes=bbox_targets, labels=metric_cls_targets),
            'metric_prediction': dict(boxes=bbox_pred, scores=scores, labels=metric_cls_pred)    
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
