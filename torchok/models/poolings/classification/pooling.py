import torch
import torch.nn.functional as F

from torchok.constructor import POOLINGS
from torchok.models.base import BaseModel


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def select_adaptive_pool2d(x, pool_type='avg', flatten: bool = True, output_size=1):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avg':
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == 'avgmax':
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == 'catavgmax':
        x = adaptive_catavgmax_pool2d(x, output_size)
    elif pool_type == 'max':
        x = F.adaptive_max_pool2d(x, output_size)
    else:
        assert False, 'Invalid pool type: %s' % pool_type

    if flatten:
        x = x.flatten(1)

    return x


@POOLINGS.register_class
class Pooling(BaseModel):
    def __init__(self, in_channels: int, pooling_type: str = 'avg', output_size=1):
        self.output_size = output_size
        self.pooling_type = pooling_type
        out_channels = in_channels if pooling_type != 'catavgmax' else 2 * in_channels
        super().__init__(in_channels, out_channels)

    def forward(self, x):
        return select_adaptive_pool2d(x, self.pooling_type, self.output_size)
