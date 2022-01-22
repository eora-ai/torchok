# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch.nn as nn

from .base_heads import AnchorHead
from src.models.layers.conv_bn_act import ConvBnAct
from src.registry import HEADS, DETECTION_HEADS


@HEADS.register_class
@DETECTION_HEADS.register_class
class RetinaHead(AnchorHead):
    """Anchor-based head for RetinaNet"""
    def __init__(self, in_features: int, num_classes: int, num_base_priors: Union[int, List[int]],
                 stacked_convs: int = 4, feat_channels: int = 256):
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels

        super().__init__(in_features=in_features,
                         num_classes=num_classes,
                         num_base_priors=num_base_priors,
                         use_sigmoid_cls=True)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_features if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvBnAct(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.ReLU))
            self.reg_convs.append(
                ConvBnAct(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_layer=nn.BatchNorm2d,
                    act_layer=nn.ReLU))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            3,
            padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        return cls_score, bbox_pred
