# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List
from fractions import Fraction as Fr

import torch
import torch.nn.functional as F
from mmcv import ConfigDict
from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32
from mmdet.core import (bbox_cxcywh_to_xyxy, build_assigner, build_sampler, reduce_mean)
from mmdet.models.dense_heads import detr_head
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils import build_transformer
from omegaconf import OmegaConf, DictConfig
from torch import Tensor

from torchok.constructor import HEADS
from torchok.losses.base import JointLoss


@HEADS.register_module()
class DETRHead(detr_head.DETRHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(self,
                 joint_loss,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 bg_cls_weight=0.1,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        kwargs['norm_cfg'] = transformer
        kwargs['conv_cfg'] = positional_encoding
        kwargs['train_cfg'] = train_cfg
        kwargs['test_cfg'] = test_cfg
        kwargs['init_cfg'] = init_cfg

        for k, v in kwargs.items():
            if isinstance(v, DictConfig):
                kwargs[k] = ConfigDict(OmegaConf.to_container(v, resolve=True))

        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = joint_loss['loss_cls'].class_weight
        if class_weight is not None and (self.__class__ is DETRHead):
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last index
            class_weight[num_classes] = bg_cls_weight
            joint_loss['loss_cls'].class_weight = class_weight
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            if 'assigner' not in train_cfg:
                raise ValueError("assigner should be provided when train_cfg is set.")
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            self.sampler = build_sampler(dict(type='PseudoSampler'), context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        num_feats = positional_encoding['num_feats']
        if num_feats * 2 != self.embed_dims:
            raise ValueError(
                f'embed_dims should be exactly 2 times of num_feats. Found {self.embed_dims} and {num_feats}.')
        self._init_layers()

        self.init_weights()

    def forward(self, features, img_metas, **kwargs):
        """Forward function.

        Args:
            features (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should include background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        features = features[-1]

        batch_size = features.size(0)
        input_img_h, input_img_w, _ = img_metas[0]['img_shape']
        masks = features.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_meta = img_metas[img_id]
            img_h, img_w = self.get_fit_image_size(img_meta['img_shape'], img_meta['orig_img_shape'])
            masks[img_id, :img_h, :img_w] = 0

        x = self.input_proj(features)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight, pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    @staticmethod
    def get_fit_image_size(fit_size, image_size):
        h, w = fit_size[:2]
        aspect_ratio = Fr(h, w)

        im_h, im_w = image_size[:2]
        image_aspect_ratio = Fr(im_h, im_w)

        den = (max if im_h > im_w else min)(im_h, im_w)
        en = h if image_aspect_ratio >= aspect_ratio else w
        scale = en / den

        return im_h * scale, im_w * scale

    @staticmethod
    def format_dict(head_output):
        return dict(zip(['all_cls_scores', 'all_bbox_preds'], head_output))

    @force_fp32(apply_to=('all_cls_scores', 'all_bbox_preds'))
    def loss(self, joint_loss: JointLoss,
             all_cls_scores,
             all_bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             **kwargs):
        """Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            joint_loss: An instance of JointLoss class.
            all_cls_scores (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses_cls = []
        losses_bbox = []
        losses_iou = []

        for cls_scores, bbox_preds in zip(all_cls_scores, all_bbox_preds):
            num_imgs = cls_scores.size(0)
            cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
            bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                               gt_bboxes_list, gt_labels_list, img_metas)
            (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
             num_total_pos, num_total_neg) = cls_reg_targets
            labels = torch.cat(labels_list, 0)
            label_weights = torch.cat(label_weights_list, 0)
            bbox_targets = torch.cat(bbox_targets_list, 0)
            bbox_weights = torch.cat(bbox_weights_list, 0)

            # classification loss
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
            # construct weighted avg_factor to match with the official DETR repo
            cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
            if self.sync_cls_avg_factor:
                cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
            cls_avg_factor = max(cls_avg_factor, 1)

            # Compute the average number of gt boxes across all gpus, for
            # normalization purposes
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

            # construct factors used for rescale bboxes
            factors = []
            for bbox_pred, img_meta in zip(bbox_preds, img_metas):
                img_h, img_w, _ = img_meta['img_shape']
                factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0).repeat(bbox_pred.size(0), 1)
                factors.append(factor)
            factors = torch.cat(factors, 0)

            # DETR regress the relative position of boxes (cxcywh) in the image,
            # thus the learning target is normalized by the image size. So here
            # we need to re-scale them for calculating IoU loss
            bbox_preds = bbox_preds.reshape(-1, 4)
            bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
            bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

            # regression IoU loss, by default GIoU loss
            loss_iou = self.loss_iou(bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

            # regression L1 loss
            loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

            joint_loss(
                cls_scores=cls_scores,
                labels=labels,
                label_weights=label_weights,
                cls_avg_factor=cls_avg_factor,
                bboxes=bboxes,
                bboxes_gt=bboxes_gt,
                bbox_weights=bbox_weights,
                bbox_preds=bbox_preds,
                bbox_targets=bbox_targets,
            )

            return loss_cls, loss_bbox, loss_iou

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers

        zi = enumerate(zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]))
        for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in zi:
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
        return loss_dict

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list: torch.Tensor,
                   all_bbox_preds_list: torch.Tensor,
                   img_metas: List[Dict],
                   **kwargs):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list: Classification outputs for the last feature level.
                Each is a 4D-tensor with shape [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list: Sigmoid regression outputs the last feature level.
                Each is a 4D-tensor with normalized coordinate format (cx, cy, w, h)
                and shape [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): List of image meta information.

        Returns:
            list[Dict[str, Tensor]]: Each item is adict with two items. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE by default only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1]
        bbox_preds = all_bbox_preds_list[-1]

        result = []
        for img_id, img_meta in enumerate(img_metas):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            det_bboxes, det_labels = self._get_bboxes_single(cls_score, bbox_pred, img_meta)
            result.append(dict(bboxes=det_bboxes, labels=det_labels))

        return result

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_meta,
                           **kwargs):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_meta (dict): List of image meta information.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]
        img_shape = img_meta['img_shape'][-2:]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels
