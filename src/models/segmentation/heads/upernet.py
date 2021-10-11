from functools import partial

import torch
import torch.nn as nn

from src.registry import SEGMENTATION_HEADS, HEADS
from .hrnet import SpatialGather_Module, SpatialOCR
from ..modules import ConvBnRelu

BatchNorm2d = nn.BatchNorm2d
resize = partial(torch.nn.functional.interpolate, mode='bilinear', align_corners=False)


@HEADS.register_class
@SEGMENTATION_HEADS.register_class
class UPerNet(nn.Module):
    has_ocr = False

    def __init__(self, num_classes, encoder_channels, drop=0.1,
                 pool_scales=(1, 2, 3, 6), fpn_dim=256, do_interpolate=True):
        super(UPerNet, self).__init__()
        self.encoder_channels = encoder_channels
        self.do_interpolate = do_interpolate
        self.num_classes = num_classes
        self.drop = drop

        # PPM Module
        ppm_pooling = []
        ppm_conv = []

        *encoder_channels, channels = encoder_channels
        for scale in pool_scales:
            ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            ppm_conv.append(ConvBnRelu(channels, 512, 1, bias=False))

        self.ppm_pooling = nn.ModuleList(ppm_pooling)
        self.ppm_conv = nn.ModuleList(ppm_conv)
        self.ppm_last_conv = ConvBnRelu(channels + len(pool_scales) * 512, fpn_dim, 3, 1)

        # FPN Module
        fpn_in = [ConvBnRelu(fpn_inplane, fpn_dim, 1, bias=False) for fpn_inplane in encoder_channels[::-1]]
        self.fpn_in = nn.ModuleList(fpn_in)

        fpn_out = [ConvBnRelu(fpn_dim, fpn_dim, 3, 1) for i in encoder_channels]  # skip the top layer
        self.fpn_out = nn.ModuleList(fpn_out)

        self.dropout = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        self.conv_last = nn.Sequential(
            ConvBnRelu((len(encoder_channels) + 1) * fpn_dim, fpn_dim, 3, 1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

        self.init_weights(self)

    def forward(self, features):
        input_image, *features, feat_last = features

        h = feat_last.size(2)
        w = feat_last.size(3)
        ppm_out = [feat_last]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(resize(pool_scale(feat_last), (h, w))))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i, feat in enumerate(features[::-1]):
            feat_x = self.fpn_in[i](feat)  # lateral branch

            # top-down branch
            f = feat_x + resize(f, size=feat_x.size()[2:])

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]] + [resize(feat, output_size) for feat in fpn_feature_list[1:]]
        fusion_out = torch.cat(fusion_list, 1)

        fusion_out = self.dropout(fusion_out)
        x = self.conv_last(fusion_out)
        if self.do_interpolate:
            x = resize(x, size=input_image.shape[2:])
        if self.num_classes == 1:
            x = x[:, 0]

        return x

    @staticmethod
    def init_weights(module):
        for m in module.modules():

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


@HEADS.register_class
@SEGMENTATION_HEADS.register_class
class UPerNetOCR(UPerNet):
    has_ocr = True

    def __init__(self, num_classes, encoder_channels, ocr_mid_channels=64, ocr_key_channels=64,
                 pool_scales=(1, 2, 3, 6), fpn_dim=256, do_interpolate=True):
        super(UPerNetOCR, self).__init__(num_classes=num_classes, encoder_channels=encoder_channels, drop=0,
                                         pool_scales=pool_scales, fpn_dim=fpn_dim, do_interpolate=do_interpolate)
        del self.dropout

        encoder_channels = fpn_dim * len(encoder_channels)
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

        self.init_weights(self)

    def forward(self, features):
        input_image, *features, feat_last = features

        h = feat_last.size(2)
        w = feat_last.size(3)
        ppm_out = [feat_last]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(resize(pool_scale(feat_last), (h, w))))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i, feat in enumerate(features[::-1]):
            feat_x = self.fpn_in[i](feat)  # lateral branch

            # top-down branch
            f = feat_x + resize(f, size=feat_x.size()[2:])

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]] + [resize(feat, output_size) for feat in fpn_feature_list[1:]]
        fusion_out = torch.cat(fusion_list, 1)

        out_aux = self.conv_last(fusion_out)
        feats = self.conv3x3_ocr(fusion_out)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        feats = self.last_reduction(feats)
        feats = resize(feats, input_image.shape[2:])
        out = self.cls_head(feats)
        out_aux = resize(out_aux, size=input_image.shape[2:])

        if self.num_classes == 1:
            out = out[:, 0]
            out_aux = out_aux[:, 0]

        if self.training:
            return out, out_aux
        else:
            return out
