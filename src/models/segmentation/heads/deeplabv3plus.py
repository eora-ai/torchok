# ------------------------------------------------------------------------------
# DeepLabV3+ decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

from src.registry import SEGMENTATION_HEADS, HEADS
from ..modules import ASPP, ConvBnRelu, stacked_conv

__all__ = ["DeepLabV3Plus"]


@HEADS.register_class
@SEGMENTATION_HEADS.register_class
class DeepLabV3Plus(nn.Module):
    has_ocr = False

    def __init__(self, num_classes, encoder_channels, decoder_channels=256, feature_key=-1,
                 low_level_key=-4, low_level_channels_project=48, atrous_rates=(6, 12, 18),
                 do_interpolate=True):
        super(DeepLabV3Plus, self).__init__()
        self.num_classes = num_classes
        self.aspp = ASPP(encoder_channels[feature_key], out_channels=decoder_channels, atrous_rates=atrous_rates)
        self.feature_key = feature_key
        self.low_level_key = low_level_key
        self.do_interpolate = do_interpolate
        # Transform low-level feature
        self.project = ConvBnRelu(encoder_channels[low_level_key],
                                  low_level_channels_project, 1, bias=False)
        # Fuse
        self.fuse = stacked_conv(
            decoder_channels + low_level_channels_project,
            decoder_channels,
            kernel_size=3,
            padding=1,
            num_stack=2,
            conv_type='depthwise_separable_conv'
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, 1)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        input_image, *features = features

        low = features[self.low_level_key]
        x = features[self.feature_key]
        x = self.aspp(x)
        # low-level feature
        low = self.project(low)
        x = F.interpolate(x, size=low.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low), dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        if self.do_interpolate:
            x = F.interpolate(x, size=input_image.shape[2:],
                              mode='bilinear', align_corners=False)
        if self.num_classes == 1:
            x = x[:, 0]
        return x
