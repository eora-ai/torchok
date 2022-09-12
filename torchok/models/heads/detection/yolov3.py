import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (bias_init_with_prob, constant_init, ConvModule, is_norm, normal_init)
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from .base import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
from mmdet.models.dense_heads import yolo_head
from torch import Tensor

from torchok.constructor import HEADS


# from functools import partial
# from typing import Tuple, Union


@HEADS.register_class
class YOLOV3Head(yolo_head.YOLOV3Head):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, **kwargs):
        super(YOLOV3Head, self).__init__(**kwargs)
        self.init_weights()

    @staticmethod
    def format_dict(head_output):
        return dict(pred_maps=head_output)

    @force_fp32(apply_to=('pred_maps',))
    def get_bboxes(self, pred_maps, with_nms=True):
        """Transform network output for a batch into bbox predictions. It has
        been accelerated since PR #5991.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg

        num_imgs = len(pred_maps[0])
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]

        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)
        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.bbox_coder.decode(flatten_anchors,
                                                flatten_bbox_preds,
                                                flatten_strides.unsqueeze(-1))

        if with_nms and (flatten_objectness.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0,))

        padding = flatten_bboxes.new_zeros(num_imgs, flatten_bboxes.shape[1], 1)
        flatten_cls_scores = torch.cat([flatten_cls_scores, padding], dim=-1)

        det_results = []
        for (bboxes, scores, objectness) in zip(flatten_bboxes,
                                                flatten_cls_scores,
                                                flatten_objectness):
            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg['score_thr'],
                dict(cfg['nms']),
                cfg['max_per_img'],
                score_factors=objectness)
            det_results.append(dict(bboxes=det_bboxes, labels=det_labels))
        return det_results

    def prepare_loss(self, pred_maps, gt_bboxes, gt_labels):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box


        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(gt_bboxes)
        device = pred_maps[0].device

        featmap_sizes = [pred_maps[i].shape[-2:] for i in range(self.num_levels)]
        mlvl_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(num_imgs):
            responsible_flags = self.prior_generator.responsible_flags(featmap_sizes, gt_bboxes[img_id], device)
            responsible_flag_list.append(responsible_flags)

        target_maps_list, neg_maps_list = self.get_targets(anchor_list, responsible_flag_list, gt_bboxes, gt_labels)

        result = []
        for args in zip(pred_maps, target_maps_list, neg_maps_list):
            result.append(self.prepare_loss_single(*args))

        return result

    def prepare_loss_single(self, pred_map, target_map, neg_map):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        pred_map = pred_map.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_attrib)
        neg_mask = neg_map.float()
        pos_mask = target_map[..., 4]
        pos_and_neg_mask = neg_mask + pos_mask

        if torch.max(pos_and_neg_mask) > 1.:
            warnings.warn('There is overlap between pos and neg sample.')
            pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)

        prep_result = dict(
            pred_xy=pred_map[..., :2],
            pred_wh=pred_map[..., 2:4],
            pred_conf=pred_map[..., 4],
            pred_label=pred_map[..., 5:],

            target_xy=target_map[..., :2],
            target_wh=target_map[..., 2:4],
            target_conf=target_map[..., 4],
            target_label=target_map[..., 5:],

            pos_mask=pos_mask.unsqueeze(dim=-1),
            pos_and_neg_mask=pos_and_neg_mask
        )

        return prep_result
