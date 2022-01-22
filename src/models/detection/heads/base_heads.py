# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, List
import torch.nn as nn

from src.models.detection.utils import multi_apply
from src.registry import HEADS, DETECTION_HEADS


@HEADS.register_class
@DETECTION_HEADS.register_class
class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        in_features: Number of channels in the input feature map.
        num_classes: Number of categories excluding the background category.
        use_sigmoid_cls: If True, then number of classes for the classification branch is exactly equal to num_classes
        (sigmoid can be applied over the output). Otherwise, classification branch will have num_classes + 1
        outputs where one bin is reserved for background class (softmax can be applied over the output)
    """
    def __init__(self, in_features: int, num_classes: int, num_base_priors: Union[int, List[int]],
                 use_sigmoid_cls: bool = False):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        if use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = num_base_priors
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_features,
                                  self.num_base_priors * self.cls_out_channels,
                                  1)
        self.conv_reg = nn.Conv2d(self.in_features, self.num_base_priors * 4,
                                  1)

    def forward_single(self, x):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)

        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: A tuple of classification scores and bbox prediction.
                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        return multi_apply(self.forward_single, feats)
