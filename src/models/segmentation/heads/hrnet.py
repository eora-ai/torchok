"""
This HRNet implementation is modified from the following repository:
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import ConvBnAct
from src.registry import SEGMENTATION_HEADS, HEADS

resize = partial(F.interpolate, mode='bilinear', align_corners=False)
ConvBnRelu = partial(ConvBnAct, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)
BN_MOMENTUM = 0.1

__all__ = ['HRNetOCR']


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class ObjectAttentionBlock(nn.Module):
    """
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1):
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

    def forward(self, x, proxy):
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
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class SpatialOCR(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels,
                                                         key_channels,
                                                         scale)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            ConvBnRelu(_in_channels, out_channels, kernel_size=1),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


@HEADS.register_class
@SEGMENTATION_HEADS.register_class
class HRNetOCR(nn.Module):
    has_ocr = True

    def __init__(self, num_classes, encoder_channels, ocr_mid_channels=128,
                 ocr_key_channels=64, **kwargs):
        super(HRNetOCR, self).__init__()
        encoder_channels = sum(encoder_channels)
        self.out_channels = encoder_channels
        self.num_classes = num_classes

        self.conv3x3_ocr = ConvBnRelu(encoder_channels, ocr_mid_channels, kernel_size=3, padding=1)
        self.ocr_gather_head = SpatialGather_Module(num_classes)

        self.ocr_distri_head = SpatialOCR(in_channels=ocr_mid_channels,
                                          key_channels=ocr_key_channels,
                                          out_channels=ocr_mid_channels,
                                          scale=1,
                                          dropout=0.05,
                                          )
        self.last_reduction = ConvBnRelu(
            ocr_mid_channels,
            ocr_mid_channels // 16,
            kernel_size=1, stride=1, padding=0
        )

        self.cls_head = nn.Conv2d(ocr_mid_channels // 16, num_classes,
                                  kernel_size=1, stride=1, padding=0, bias=True)

        self.aux_head = nn.Sequential(
            ConvBnRelu(encoder_channels, encoder_channels,
                       kernel_size=1, stride=1, padding=0),
            nn.Conv2d(encoder_channels, num_classes,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.init_weights()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        input_image, *features = features
        xs = [features[0]]
        for y in features[1:]:
            xs.append(resize(y, size=features[0].shape[2:]))

        feats = torch.cat(xs, 1)
        # ocr
        out_aux = self.aux_head(feats)
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        feats = self.last_reduction(feats)
        feats = F.interpolate(feats, input_image.shape[2:], mode='bilinear', align_corners=True)
        out = self.cls_head(feats)

        if self.training:
            out_aux = F.interpolate(out_aux, input_image.shape[2:], mode='bilinear', align_corners=True)

        if self.num_classes == 1:
            out = out[:, 0]
            out_aux = out_aux[:, 0]

        if self.training:
            return out, out_aux
        else:
            return out
