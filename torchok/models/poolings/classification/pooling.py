from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

from torchok.constructor import POOLINGS
from torchok.models.base import BaseModel


@POOLINGS.register_class
class Pooling(SelectAdaptivePool2d, BaseModel):
    def __init__(self, in_channels: int, pooling_type: str = 'avg', output_size: int = 1):
        super().__init__(output_size=output_size, pool_type=pooling_type, flatten=True)
        self._in_channels = in_channels
        self._out_channels = in_channels if pooling_type != 'catavgmax' else 2 * in_channels
