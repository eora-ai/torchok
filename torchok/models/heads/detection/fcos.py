# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmcv import ConfigDict
from mmcv.runner import force_fp32
from mmdet.core import reduce_mean
from mmdet.models.dense_heads import fcos_head
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from omegaconf import OmegaConf, DictConfig
from torch import Tensor

from torchok.constructor import HEADS
from torchok.losses.base import JointLoss

INF = 1e8


@HEADS.register_class
class FCOSHead(fcos_head.FCOSHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 conv_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg

        kwargs['norm_cfg'] = norm_cfg
        kwargs['conv_cfg'] = conv_cfg
        kwargs['train_cfg'] = train_cfg
        kwargs['test_cfg'] = test_cfg
        kwargs['init_cfg'] = init_cfg

        for k, v in kwargs.items():
            if isinstance(v, DictConfig):
                kwargs[k] = ConfigDict(OmegaConf.to_container(v, resolve=True))
        AnchorFreeHead.__init__(self, num_classes, in_channels, **kwargs)

        self.init_weights()

    @staticmethod
    def format_dict(head_output):
        return dict(zip(['cls_scores', 'bbox_preds', 'centernesses'], head_output))

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self, joint_loss: JointLoss, cls_scores: List[Tensor], bbox_preds: List[Tensor],
             centernesses: List[Tensor], gt_bboxes: List[Tensor], gt_labels: List[Tensor],
             **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute loss of the head.

        Args:
            joint_loss: An instance of JointLoss class.
            cls_scores: Box scores for each scale level, each is a 4D-tensor,
                the channel number is num_points * num_classes.
            bbox_preds: Box energies / deltas for each scale level,
                each is a 4D-tensor, the channel number is num_points * 4.
            centernesses: centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes: Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels: class indices corresponding to each box

        Returns:
            Total loss, Dict of losses per eachA dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        # centerness weighted iou loss
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        pos_points = flatten_points[pos_inds]
        pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
        pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)

        return joint_loss(
            flatten_cls_scores=flatten_cls_scores.float(),
            flatten_labels=flatten_labels,
            num_pos=num_pos,
            pos_decoded_bbox_preds=pos_decoded_bbox_preds.float(),
            pos_decoded_target_preds=pos_decoded_target_preds.float(),
            pos_centerness_targets=pos_centerness_targets.float(),
            centerness_denorm=centerness_denorm,
            pos_centerness=pos_centerness.float()
        )

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, image_shape, **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            image_shape (Tuple[int, int]): size of the input image.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        pseudo_meta = [dict(img_shape=image_shape, scale_factor=1) for i in range(len(cls_scores[0]))]
        result = super().get_bboxes(cls_scores=cls_scores, bbox_preds=bbox_preds,
                                    img_metas=pseudo_meta, **kwargs)
        result = [dict(bboxes=bboxes, labels=labels) for bboxes, labels in result]
        return result
