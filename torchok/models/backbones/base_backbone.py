from torch import Tensor
from abc import ABC
from typing import Tuple, List

from timm.models.features import FeatureHooks

from torchok.models.base import BaseModel


class BaseBackbone(BaseModel, ABC):
    """Base model for TorchOk Backbones"""

    def create_hooks(self):
        """Crete hooks for intermediate encoder features based on model's feature info.
        """
        self.stage_names = [h['module'] for h in self.feature_info]
        self._out_encoder_channels = [h['num_chs'] for h in self.feature_info]
        for h in self.feature_info:
            # default hook type denoted here as a `` that cause error in FeatureHooks
            # remove it to use default hook type from FeatureHooks
            if 'hook_type' in h and not h['hook_type']:
                del h['hook_type']
        self.feature_hooks = FeatureHooks(self.feature_info, self.named_modules())

    def forward_features(self, x: Tensor) -> List[Tensor]:
        """Forward method for getting backbone feature maps.
           They are mainly used for segmentation and detection tasks.
        """
        last_features = self(x)
        backbone_features = self.feature_hooks.get_output(x.device)
        backbone_features = list(backbone_features.values())
        return [x] + backbone_features

    @property
    def out_encoder_channels(self) -> Tuple[int]:
        """Number of output feature channels - channels after forward_features method."""
        if self._out_encoder_channels is None:
            raise ValueError('TorchOk Backbones must have self._out_feature_channels attribute.')
        return tuple(self._out_encoder_channels)
