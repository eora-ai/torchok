""" MobileNet V3
Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py
A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.


Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by / Copyright 2019, Ross Wightman
"""
from functools import partial

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.efficientnet_blocks import SqueezeExcite
from timm.models.efficientnet_builder import BN_EPS_TF_DEFAULT, decode_arch_def, efficientnet_init_weights, \
    EfficientNetBuilder, resolve_act_layer, resolve_bn_args, round_channels
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import create_conv2d

from torchok.constructor import BACKBONES
from torchok.models.backbones import BaseBackbone


def _cfg(url='', **kwargs):
    return {
        'url': url, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mobilenetv3_large_075': _cfg(url=''),
    'mobilenetv3_large_100': _cfg(
        interpolation='bicubic',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'), # noqa
    'mobilenetv3_large_100_miil': _cfg(
        interpolation='bilinear', mean=(0, 0, 0), std=(1, 1, 1),
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/mobilenetv3_large_100_1k_miil_78_0.pth'), # noqa
    'mobilenetv3_large_100_miil_in21k': _cfg(
        interpolation='bilinear', mean=(0, 0, 0), std=(1, 1, 1),
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/mobilenetv3_large_100_in21k_miil.pth', # noqa
        num_classes=11221),

    'mobilenetv3_small_050': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_050_lambc-4b7bbe87.pth', # noqa
        interpolation='bicubic'),
    'mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_075_lambc-384766db.pth', # noqa
        interpolation='bicubic'),
    'mobilenetv3_small_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_100_lamb-266a294c.pth', # noqa
        interpolation='bicubic'),

    'mobilenetv3_rw': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth', # noqa
        interpolation='bicubic'),

    'tf_mobilenetv3_large_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth', # noqa
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth', # noqa
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth', # noqa
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth', # noqa
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth', # noqa
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth', # noqa
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'fbnetv3_b': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_b_224-ead5d2a1.pth', # noqa
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_d_224-c98bce42.pth', # noqa
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_g': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_g_240-0b1df83b.pth', # noqa
        input_size=(3, 240, 240), test_input_size=(3, 288, 288), crop_pct=0.95, pool_size=(8, 8)),

    "lcnet_035": _cfg(),
    "lcnet_050": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_050-f447553b.pth', # noqa
        interpolation='bicubic',
    ),
    "lcnet_075": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_075-318cad2c.pth', # noqa
        interpolation='bicubic',
    ),
    "lcnet_100": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_100-a929038c.pth', # noqa
        interpolation='bicubic',
    ),
    "lcnet_150": _cfg(),
}


class MobileNetV3(BaseBackbone):
    """ MobiletNet-V3

    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.

    Paper: `Searching for MobileNetV3` - https://arxiv.org/abs/1905.02244

    Other architectures utilizing MobileNet-V3 efficient head that are supported by this impl include:
      * HardCoRe-NAS - https://arxiv.org/abs/2102.11646 (defn in hardcorenas.py uses this class)
      * FBNet-V3 - https://arxiv.org/abs/2006.02049
      * LCNet - https://arxiv.org/abs/2109.15099
    """

    def __init__(self, block_args, in_channels=3, stem_size=16, fix_stem=False, num_features=1280, pad_type='',
                 act_layer=None, norm_layer=None, se_layer=None, se_from_exp=True, round_chs_fn=round_channels,
                 output_stride=32, drop_path_rate=0.):
        super(MobileNetV3, self).__init__(in_channels=in_channels, out_channels=num_features)
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.num_features = num_features

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_channels, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer,
            drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features

        self.create_hooks()
        self.init_weights()

    @torch.jit.ignore
    def init_weights(self):
        efficientnet_init_weights(self)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x

    def get_stages(self, stage: int) -> nn.Module:
        """Return modules corresponding the given model stage and all previous stages.
        For example, `0` must stand for model stem. `1` must stand for models stem and
        the first global layer of the model (`layer1` in the resnet), etc.

        Args:
            stage: index of the models stage.
        """
        output = [self.conv_stem, self.bn1, self.act1]
        return nn.ModuleList(output + list(self.blocks[:stage]))


def _create_mnv3(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(MobileNetV3, variant, pretrained, pretrained_strict=False,
                                kwargs_filter=('num_classes', 'global_pool', 'in_chans', 'head_bias'), **kwargs)


def _gen_mobilenet_v3_rw(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid'),
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16'],
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                # stage 2, 56x56 in
                ['ir_r3_k3_s2_e3_c40'],
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112'],
                # stage 5, 14x14in
                ['ir_r3_k3_s2_e6_c160'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer = resolve_act_layer(kwargs, 'hard_swish')
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                # stage 2, 56x56 in
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                # stage 5, 14x14in
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU, rd_round_fn=round_channels)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        fix_stem=channel_multiplier < 0.75,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_fbnetv3(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ FBNetV3
    Paper: `FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining`
        - https://arxiv.org/abs/2006.02049
    FIXME untested, this is a preliminary impl of some FBNet-V3 variants.
    """
    vl = variant.split('_')[-1]
    if vl in ('a', 'b'):
        stem_size = 16
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k5_s2_e4_c24', 'ir_r3_k5_s1_e2_c24'],
            ['ir_r1_k5_s2_e5_c40_se0.25', 'ir_r4_k5_s1_e3_c40_se0.25'],
            ['ir_r1_k5_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c120_se0.25', 'ir_r5_k5_s1_e3_c120_se0.25'],
            ['ir_r1_k3_s2_e6_c184_se0.25', 'ir_r5_k5_s1_e4_c184_se0.25', 'ir_r1_k5_s1_e6_c224_se0.25'],
            ['cn_r1_k1_s1_c1344'],
        ]
    elif vl == 'd':
        stem_size = 24
        arch_def = [
            ['ds_r2_k3_s1_e1_c16'],
            ['ir_r1_k3_s2_e5_c24', 'ir_r5_k3_s1_e2_c24'],
            ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r4_k3_s1_e3_c40_se0.25'],
            ['ir_r1_k3_s2_e5_c72', 'ir_r4_k3_s1_e3_c72'],
            ['ir_r1_k3_s1_e5_c128_se0.25', 'ir_r6_k5_s1_e3_c128_se0.25'],
            ['ir_r1_k3_s2_e6_c208_se0.25', 'ir_r5_k5_s1_e5_c208_se0.25', 'ir_r1_k5_s1_e6_c240_se0.25'],
            ['cn_r1_k1_s1_c1440'],
        ]
    elif vl == 'g':
        stem_size = 32
        arch_def = [
            ['ds_r3_k3_s1_e1_c24'],
            ['ir_r1_k5_s2_e4_c40', 'ir_r4_k5_s1_e2_c40'],
            ['ir_r1_k5_s2_e4_c56_se0.25', 'ir_r4_k5_s1_e3_c56_se0.25'],
            ['ir_r1_k5_s2_e5_c104', 'ir_r4_k3_s1_e3_c104'],
            ['ir_r1_k3_s1_e5_c160_se0.25', 'ir_r8_k5_s1_e3_c160_se0.25'],
            ['ir_r1_k3_s2_e6_c264_se0.25', 'ir_r6_k5_s1_e5_c264_se0.25', 'ir_r2_k5_s1_e6_c288_se0.25'],
            ['cn_r1_k1_s1_c1728'],
        ]
    else:
        raise NotImplementedError
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, round_limit=0.95)
    se_layer = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=round_chs_fn)
    act_layer = resolve_act_layer(kwargs, 'hard_swish')
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1984,
        head_bias=False,
        stem_size=stem_size,
        round_chs_fn=round_chs_fn,
        se_from_exp=False,
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        se_layer=se_layer,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


def _gen_lcnet(variant, channel_multiplier=1.0, pretrained=False, **kwargs):
    """ LCNet
    Essentially a MobileNet-V3 crossed with a MobileNet-V1

    Paper: `PP-LCNet: A Lightweight CPU Convolutional Neural Network` - https://arxiv.org/abs/2109.15099

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['dsa_r1_k3_s1_c32'],
        # stage 1, 112x112 in
        ['dsa_r2_k3_s2_c64'],
        # stage 2, 56x56 in
        ['dsa_r2_k3_s2_c128'],
        # stage 3, 28x28 in
        ['dsa_r1_k3_s2_c256', 'dsa_r1_k5_s1_c256'],
        # stage 4, 14x14in
        ['dsa_r4_k5_s1_c256'],
        # stage 5, 14x14in
        ['dsa_r2_k5_s2_c512_se0.25'],
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite, gate_layer='hard_sigmoid', force_act_layer=nn.ReLU),
        num_features=1280,
        **kwargs,
    )
    model = _create_mnv3(variant, pretrained, **model_kwargs)
    return model


@BACKBONES.register_class
def mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def mobilenetv3_large_100_miil(pretrained=False, **kwargs):
    """ MobileNet V3
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model = _gen_mobilenet_v3('mobilenetv3_large_100_miil', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def mobilenetv3_large_100_miil_in21k(pretrained=False, **kwargs):
    """ MobileNet V3, 21k pretraining
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model = _gen_mobilenet_v3('mobilenetv3_large_100_miil_in21k', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def mobilenetv3_small_050(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_050', 0.50, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    model = _gen_mobilenet_v3('mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def mobilenetv3_rw(pretrained=False, **kwargs):
    """ MobileNet V3 """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    model = _gen_mobilenet_v3_rw('mobilenetv3_rw', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def tf_mobilenetv3_large_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def tf_mobilenetv3_large_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def tf_mobilenetv3_large_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def tf_mobilenetv3_small_075(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def tf_mobilenetv3_small_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def tf_mobilenetv3_small_minimal_100(pretrained=False, **kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_minimal_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def fbnetv3_b(pretrained=False, **kwargs):
    """ FBNetV3-B """
    model = _gen_fbnetv3('fbnetv3_b', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def fbnetv3_d(pretrained=False, **kwargs):
    """ FBNetV3-D """
    model = _gen_fbnetv3('fbnetv3_d', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def fbnetv3_g(pretrained=False, **kwargs):
    """ FBNetV3-G """
    model = _gen_fbnetv3('fbnetv3_g', pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def lcnet_035(pretrained=False, **kwargs):
    """ PP-LCNet 0.35"""
    model = _gen_lcnet('lcnet_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def lcnet_050(pretrained=False, **kwargs):
    """ PP-LCNet 0.5"""
    model = _gen_lcnet('lcnet_050', 0.5, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def lcnet_075(pretrained=False, **kwargs):
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def lcnet_100(pretrained=False, **kwargs):
    """ PP-LCNet 1.0"""
    model = _gen_lcnet('lcnet_100', 1.0, pretrained=pretrained, **kwargs)
    return model


@BACKBONES.register_class
def lcnet_150(pretrained=False, **kwargs):
    """ PP-LCNet 1.5"""
    model = _gen_lcnet('lcnet_150', 1.5, pretrained=pretrained, **kwargs)
    return model
