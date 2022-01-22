from pydantic import BaseModel

import torch
from torch.nn import functional as F

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.models.detection.utils import images_to_levels, multi_apply
from src.registry import TASKS, DETECTION_NECKS, DETECTION_HEADS, DETECTION_HATS
from .base_task import BaseTask


class DetectionParams(BaseModel):
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    backbone_name: str
    backbone_params: dict = {}
    neck_name: str
    neck_params: dict = {}
    head_name: str
    head_params: dict = {}
    hat_name: str
    hat_params: dict = {}
    freeze_backbone: bool = False
    skip_loss_on_eval: bool = True


@TASKS.register_class
class DetectionTask(BaseTask):
    config_parser = DetectionParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)

        # create backbone
        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params)

        # create neck
        self.neck = DETECTION_NECKS.get(self.params.neck_name)(**self.params.neck_params)

        # create head
        self.head = DETECTION_HEADS.get(self.params.head_name)(**self.params.head_params)

        # create hat
        self.hat = DETECTION_HATS.get(self.params.hat_name)(**self.params.hat_params)

    def forward(self, x):
        with torch.set_grad_enabled(not self.params.freeze_backbone and self.training):
            _, features = self.backbone.forward_backbone_features(x)
            features = features[2:]
        features = self.neck(features)
        cls_scores, bbox_preds = self.head(features)

        return cls_scores, bbox_preds

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

    def _parse_batch(self, batch):
        input_data = batch['input']
        n_images = len(input_data)
        gt_bboxes = [batch['target_bboxes'][i][:batch['bbox_count'][i]] for i in range(n_images)]
        gt_labels = [batch['target_classes'][i][:batch['bbox_count'][i]] for i in range(n_images)]
        img_metas = [{
            'pad_shape': batch['pad_shape'][i],
            'img_shape': batch['img_shape'][i],
            'scale_factor': batch['scale_factor'][i]
        } for i in range(n_images)]

        return input_data, gt_bboxes, gt_labels, img_metas

    def forward_train(self, batch):
        input_data, gt_bboxes, gt_labels, img_metas = self._parse_batch(batch)

        cls_scores, bbox_preds = self.forward(input_data)
        output = self.prepare_loss_inputs(cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas)

        return output

    def prepare_loss_inputs(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.hat.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.hat.get_anchors(featmap_sizes, img_metas, device=device)
        cls_reg_targets = self.hat.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_labels_list=gt_labels)
        if cls_reg_targets is None:
            return torch.tensor(0.)

        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,\
        num_total_pos, num_total_neg = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        cls_score_per_level, labels_per_level, label_weights_per_level, bbox_pred_per_level,\
        bbox_targets_per_level, bbox_weights_per_level = multi_apply(
            self._prepare_loss_inputs_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list)

        cls_score = torch.cat(cls_score_per_level)
        labels = torch.cat(labels_per_level)
        label_weights = torch.cat(label_weights_per_level)
        bbox_pred = torch.cat(bbox_pred_per_level)
        bbox_targets = torch.cat(bbox_targets_per_level)
        bbox_weights = torch.cat(bbox_weights_per_level)

        # FIXME: label weights aren't used in loss functions
        output = {
            "cls_score": cls_score,
            "labels": labels,
            # "label_weights": label_weights,
            "bbox_pred": bbox_pred,
            "bbox_targets": bbox_targets,
            # "bbox_weights": bbox_weights
        }

        return output

    def _prepare_loss_inputs_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                                    bbox_targets, bbox_weights):

        # classification loss
        labels = labels.reshape(-1)
        # TODO: will not work when softmax is used
        labels = F.one_hot(labels, self.hat.cls_out_channels).float()
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.hat.cls_out_channels)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.hat.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.hat.bbox_coder.decode(anchors, bbox_pred)

        return cls_score, labels, label_weights, bbox_pred, bbox_targets, bbox_weights

    def training_step(self, batch, batch_idx):
        output = self.forward_train(batch)
        loss = self.criterion(**output)
        # self.metric_manager.update('train', **output)

        return loss

    def forward_eval(self, batch, score_thr=0.0, rescale=False):
        input_data, gt_bboxes, gt_labels, img_metas = self._parse_batch(batch)

        cls_scores, bbox_preds = self.forward(input_data)
        det_bboxes, det_labels = self.hat.get_bboxes(cls_scores, bbox_preds,
                                                     img_metas=img_metas, score_thr=score_thr,
                                                     rescale=rescale, with_nms=True)

        prediction = [[det_bboxes[i], det_labels[i]] for i in range(len(gt_bboxes))]
        target = [[gt_bboxes[i], gt_labels[i]] for i in range(len(gt_bboxes))]

        output = {
            'prediction': prediction,
            'target': target
        }

        return output

    def _eval_step(self, batch, batch_idx, mode):
        output = self.forward_eval(batch)

        if not self.params.skip_loss_on_eval:
            loss = self.criterion(**output)
            self.metric_manager.update(mode, **output)

            return loss
        else:
            self.metric_manager.update(mode, **output)

            return torch.tensor(0.)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, 'test')
