import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import SEGMENTATION_HEADS, HEADS
from ..modules import ConvBnRelu, Attention


@HEADS.register_class
@SEGMENTATION_HEADS.register_class
class Unet(nn.Module):
    """Unet is a fully convolution neural network for image semantic segmentation.
    Paper: https://arxiv.org/pdf/1505.04597.

    Args:
        num_classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        n_blocks (int): number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature tensor will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
    """
    has_ocr = False

    def __init__(self, num_classes, encoder_channels, decoder_channels=(256, 128, 64, 32, 16),
                 n_blocks=5, use_batchnorm=True, attention_type=None, center=False, do_interpolate=True):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        self.do_interpolate = do_interpolate
        self.num_classes = num_classes
        self.n_blocks = n_blocks
        encoder_channels = encoder_channels[1 - n_blocks:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        self.init_weights(self)

    def forward(self, features):
        input_image, *features = features

        features = features[1 - self.n_blocks:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        segm_logits = self.classifier(x)
        if self.do_interpolate:
            segm_logits = F.interpolate(segm_logits, size=input_image.shape[2:],
                                        mode='bilinear', align_corners=False)
        if self.num_classes == 1:
            segm_logits = segm_logits[:, 0]
        return segm_logits

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


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = ConvBnRelu(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = ConvBnRelu(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if skip.size(2) != x.size(2):
                skip = F.interpolate(skip, size=(x.size(2), x.size(3)), mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = ConvBnRelu(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = ConvBnRelu(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)
