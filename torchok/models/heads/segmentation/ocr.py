"""
This HRNet implementation is modified from the following repository:
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchok.constructor import HEADS
from torchok.models.base import BaseModel
from torchok.models.modules.bricks.convbnact import ConvBnAct

resize = partial(F.interpolate, mode='bilinear', align_corners=False)
ConvBnRelu = partial(ConvBnAct, act_layer=nn.ReLU)
BN_MOMENTUM = 0.1


class SpatialGather_Module(nn.Module):
    """Aggregate the context features according to the initial predicted probability distribution.
    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, num_classes: int = 0, scale: int = 1):
        """Init ObjectAttentionBlock.
        Args:
            num_classes: Number of classes.
            scale: Scale to downsample the input feature maps (save memory cost).
        """
        super(SpatialGather_Module, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, feats: Tensor, probs: Tensor) -> Tensor:
        batch_size, c, _, _ = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    """The basic implementation for object context block."""

    def __init__(self, in_channels: int, key_channels: int, scale: int = 1):
        """Init ObjectAttentionBlock.
        Args:
            in_channels: The dimension of the input feature map.
            key_channels: Number of channels in the dimension after the key/query transform.
            scale: Scale to downsample the input feature maps (save memory cost).
        """
        super(ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            ConvBnRelu(self.in_channels, self.key_channels, kernel_size=1),
            ConvBnRelu(self.key_channels, self.key_channels, kernel_size=1)
        )
        self.f_object = nn.Sequential(
            ConvBnRelu(self.in_channels, self.key_channels, kernel_size=1),
            ConvBnRelu(self.key_channels, self.key_channels, kernel_size=1)
        )
        self.f_down = nn.Sequential(
            ConvBnRelu(self.in_channels, self.key_channels, kernel_size=1),
            ConvBnRelu(self.key_channels, self.key_channels, kernel_size=1)
        )
        self.f_up = ConvBnRelu(self.key_channels, self.in_channels, kernel_size=1)

    def forward(self, x: Tensor, proxy: Tensor) -> Tensor:
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = resize(input=context, size=(h, w))

        return context


class SpatialOCR(nn.Module):
    """Implementation of the OCR module: Aggregate the global object representation to update the
    representation for each pixel.
    """

    def __init__(self, in_channels: int, key_channels: int, out_channels: int, scale: int = 1, dropout: float = 0.1):
        """Init SpatialOCR.
        Args:
            in_channels: Number of input channels.
            key_channels: Number of channels in the dimension after the key/query transform.
            out_channels: Number of output channels.
            scale: Scale to downsample the input feature maps (save memory cost).
            dropout: Dropout probability.
        """
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels, key_channels, scale)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            ConvBnRelu(_in_channels, out_channels, kernel_size=1),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats: Tensor, proxy_feats: Tensor) -> Tensor:
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


@HEADS.register_class
class OCRSegmentationHead(BaseModel):
    """
    Implementation of HRNet segmentation head with Object-Contextual Representations for Semantic Segmentation
    from https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR.
    """

    def __init__(self, in_channels: int, num_classes: int, do_interpolate: bool = True,
                 ocr_mid_channels=128, ocr_key_channels=64):
        """Init OCRSegmentationHead.
        Args:
            in_channels: Number of channels from segmentation neck.
            num_classes: Number of segmentation classes.
            ocr_mid_channels: Number of intermediate feature channels.
            ocr_key_channels: Number of channels in the dimension after the key/query transform.
        """
        super().__init__(in_channels, num_classes)
        self.do_interpolate = do_interpolate
        self.num_classes = num_classes

        self.conv3x3_ocr = ConvBnRelu(in_channels, ocr_mid_channels, kernel_size=3, padding=1)
        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR(in_channels=ocr_mid_channels, key_channels=ocr_key_channels,
                                          out_channels=ocr_mid_channels, scale=1, dropout=0.05)

        self.last_reduction = ConvBnRelu(ocr_mid_channels, ocr_mid_channels // 16, kernel_size=1, stride=1, padding=0)

        self.aux_head = nn.Sequential(
            ConvBnRelu(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.classifier = nn.Conv2d(ocr_mid_channels // 16, num_classes, kernel_size=1)

    def forward(self, feats: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        input_image, feats = feats

        out_aux = self.aux_head(feats)
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        feats = self.last_reduction(feats)
        out = self.classifier(feats)

        if self.do_interpolate:
            out = resize(out, input_image.shape[2:])
            out_aux = resize(out_aux, input_image.shape[2:])

        if self.num_classes == 1:
            out = out[:, 0]
            out_aux = out_aux[:, 0]

        if self.training:
            return out, out_aux
        else:
            return out
