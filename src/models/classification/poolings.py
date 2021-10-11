import torch
from torch import nn, Tensor

from src.registry import POOLINGS
from ..backbones.vision_transformer import Block
from ..layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from ..layers.create_act import create_act_layer


@POOLINGS.register_class
class Pooling(nn.Module):
    def __init__(self, in_features, global_pool='fast'):
        super().__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool, flatten=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        return x


@POOLINGS.register_class
class PoolingLinear(Pooling):
    def __init__(self, in_features, out_features, global_pool='fast', bias=True):
        super().__init__(in_features=in_features, global_pool=global_pool)
        self.out_features = out_features

        feat_mult = 1 if global_pool is None else self.global_pool.feat_mult()
        self.fc = nn.Linear(self.in_features * feat_mult, self.out_features, bias=bias)
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        x = self.fc(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


@POOLINGS.register_class
class PoolingMLP(Pooling):
    def __init__(self, in_features, hidden_features, out_features, num_layers, global_pool='fast',
                 act_name='relu', act_params=None, has_norm=False, bias=True):
        super().__init__(in_features=in_features, global_pool=global_pool)
        if num_layers < 1:
            raise ValueError('num_layers must be more than one, otherwise use PoolingLinear')
        self.num_layers = num_layers
        self.out_features = out_features

        if act_params is None:
            act_params = {}

        layers = []
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features] * (num_layers - 1)
        elif isinstance(hidden_features, (tuple, list)) and len(hidden_features) != num_layers - 1:
            raise ValueError('Length of `hidden_features` must be equal to `num_layers` - 1')
        hidden_features = [in_features] + hidden_features + [out_features]

        for i in range(num_layers):
            if has_norm:
                layers.append(nn.LayerNorm(hidden_features[i]))
            layers.append(create_act_layer(act_name, **act_params))
            layers.append(nn.Linear(hidden_features[i], hidden_features[i + 1], bias=bias or (i + 1 != num_layers)))
        self.layers = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        x = self.global_pool(x)
        x = self.layers(x)
        return x


@POOLINGS.register_class
class MultiPooling(nn.Module):
    def __init__(self, in_features, poolings):
        super().__init__()
        self.in_features = in_features

        self.poolings = []
        self.out_features = []
        for pooling in poolings:
            params = pooling.get('params', {})
            output_size = params.get('output_size', [1, 1])
            pool = SelectAdaptivePool2d(pool_type=pooling['name'], flatten=True, **params)
            self.poolings.append(pool)
            self.out_features.append(in_features * output_size[0] * output_size[1] * pool.feat_mult())

    def forward(self, x: Tensor) -> Tensor:
        ys = []
        for pool in self.poolings:
            y = pool(x)
            ys.append(y)

        x = torch.cat(ys, dim=1)

        return x


@POOLINGS.register_class
class TransformerMultiImagePooling(nn.Module):
    def __init__(self, in_features, out_features, num_images_per_group, drop=0.,
                 attn_drop=0., drop_path=0., bias=True, always_apply_mi=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop = nn.Dropout(drop)
        self.num_images_per_group = num_images_per_group
        self.always_apply_mi = always_apply_mi
        self.combine_block = Block(in_features, 1, drop=drop,
                                   attn_drop=attn_drop, drop_path=drop_path)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def forward(self, x: Tensor) -> Tensor:
        bn, c, h, w = x.shape
        n = self.num_images_per_group if self.training or self.always_apply_mi else 1
        x = x.permute(0, 2, 3, 1).view(bn // n, n * h * w, c)
        x = self.combine_block(x).mean(1)

        self.drop(x)
        x = self.linear(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
