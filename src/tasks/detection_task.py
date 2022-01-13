from typing import DefaultDict
import torch
from pydantic import BaseModel

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, DETECTION_NECKS, \
    DETECTION_HEADS, DETECTION_HATS
from .base_task import BaseTask
import torch.nn as nn
from src.models.backbones.utils.hub import download_cached_file

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
                            num_csp_blocks=1
        )
        
        head_class = DETECTION_HEADS.get(self.params.head_name)
        self.head = head_class(
                            num_classes=self.params.head_params['num_classes'], 
                            in_channels=128, 
                            feat_channels=128
        )

        infer_class = DETECTION_HATS.get(self.params.hat_name)
        self.infer_module = infer_class(num_classes=self.params.head_params['num_classes'], **self.params.hat_params)
        self.num_classes = self.params.head_params['num_classes']

        self._set_pretrained_weights()

    def _set_pretrained_weights(self, url = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'):
        file = download_cached_file(url)
        loaded_state_dict = torch.load(file)['state_dict']
        model_state_dict = self.state_dict()
        # print(model_state_dict)
        loaded_keys = list(loaded_state_dict.keys())
        model_state_dict
        lol = 0
        for key, value in model_state_dict.items():
            index = key.find('.')
            module_name = key[:index]
            weight_name = key[index+1:]
            load_value = None
            for loaded_key in loaded_keys:
                index = loaded_key.find('.')
                loaded_module_name = loaded_key[:index]
                if module_name in loaded_module_name and loaded_key.endswith(weight_name):
                    load_value = loaded_state_dict[loaded_key]
                    break
            if load_value is not None and load_value.shape == value.shape:
                lol += 1
                model_state_dict[key] = load_value
        loaded = self.load_state_dict(model_state_dict)
        print('load state dict = ' + str(loaded))

    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        cls_score, bbox_pred, objectness = self.head(neck_features)
        return cls_score, bbox_pred, objectness

    def forward_valid(self, x):
        cls_score, bbox_pred, objectness = self.forward(x)
        prediction = self.infer_module.forward_infer(cls_score, bbox_pred, objectness)
        return prediction

    def forward_with_gt(self, batch):
        input_data = batch['input']
        gt_bboxes = batch['target_bboxes']
        gt_labels = batch['target_classes']

        cls_score, bbox_pred, objectness = self.forward(input_data)

        num_total_samples, \
            bbox_pred, bbox_targets,\
                obj_pred, obj_targets,\
                    cls_pred, cls_targets = self.infer_module.forward_train(
                                                            cls_score,
                                                            bbox_pred,
                                                            objectness, 
                                                            gt_bboxes=gt_bboxes,
                                                            gt_labels=gt_labels
                                                            )
        
        output = {
            'bbox_pred': bbox_pred, 
            'bbox_target': bbox_targets, 
            'obj_pred': obj_pred, 
            'obj_target': obj_targets,
            'cls_pred': cls_pred, 
            'cls_targets': cls_targets,
            }
        return output

    def training_step(self, batch, batch_idx):
        output = self.forward_with_gt(batch)
        loss = self.criterion(**output)
        # self.metric_manager.update('train', **output)
        return loss

    def validation_step(self, batch, batch_idx):
        input_data = batch['input']
        gt_bboxes = batch['target_bboxes']
        gt_labels = batch['target_classes']
        prediction = self.forward_valid(input_data)
        # prediction is list of elements 
        target = [[gt_bboxes[i], gt_labels[i]] for i in range(gt_bboxes.shape[0])]
        valid_output = {
            'target': target,
            'prediction': prediction
        }
        self.metric_manager.update('valid', **valid_output)
        return torch.tensor(0.)

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
