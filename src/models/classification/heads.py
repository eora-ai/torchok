import torch
import torch.nn.functional as F
from torch import nn, Tensor

from src.registry import HEADS, CLASSIFICATION_HEADS
from ..layers.create_act import create_act_layer


class AbstractHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        raise NotImplementedError()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


@HEADS.register_class
@CLASSIFICATION_HEADS.register_class
class LinearHead(AbstractHead):
    def __init__(self, in_features, out_features, drop_rate=0.0, bias=True, normalize=False):
        super().__init__(in_features, out_features)
        self.drop_rate = drop_rate
        self.normalize = normalize
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = self.fc(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x


@HEADS.register_class
@CLASSIFICATION_HEADS.register_class
class ClassificationHead(AbstractHead):
    def __init__(self, in_features, num_classes, drop_rate=0.0, bias=True):
        super().__init__(in_features, num_classes)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.fc = nn.Linear(in_features, num_classes, bias=bias)
        self.init_weights()

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        if self.num_classes == 1:
            x = x[:, 0]

        return x


@HEADS.register_class
@CLASSIFICATION_HEADS.register_class
class MLPHead(AbstractHead):
    def __init__(self, in_features, hidden_features, out_features, num_layers,
                 act_name='relu', act_params=None, has_norm=True, bias=True, normalize=False):
        super().__init__(in_features, out_features)

        if act_params is None:
            act_params = dict()
        self.normalize = normalize

        if num_layers < 1:
            raise ValueError('num_layers must be more than one, otherwise use ClassificationHead')
        self.num_layers = num_layers

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

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        x = self.layers(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x


@HEADS.register_class
@CLASSIFICATION_HEADS.register_class
class IdentityHead(AbstractHead):
    def __init__(self, in_features, **kwargs):
        super().__init__(in_features, in_features)

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:

        return x


@HEADS.register_class
@CLASSIFICATION_HEADS.register_class
class NormalizationHead(AbstractHead):
    def __init__(self, in_features, normalize=True, **kwargs):
        super().__init__(in_features, in_features)
        self.normalize = normalize

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x
