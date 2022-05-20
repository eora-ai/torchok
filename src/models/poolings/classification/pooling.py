from torch import nn, Tensor

from src.constructor import POOLINGS


class FastAdaptiveAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.mean((2, 3)) if self.flatten else x.mean((2, 3), keepdim=True)


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type='fast', flatten=True, **pooling_args):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = flatten
        if pool_type == 'fast':
            assert output_size == 1
            self.pool = FastAdaptiveAvgPool2d(self.flatten)
            self.flatten = False

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            x = x.flatten(1)
        return x



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

    def get_forward_output_channels(self):
        return self.out_features