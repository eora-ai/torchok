from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional

from src.models.base import BaseModel
from timm.models.features import FeatureHooks


class BaseBackbone(BaseModel, ABC):
    """Base model for TorchOk Backbones"""

    def create_hooks(self):
        self.stage_names = [i['module'] for i in self.feature_info]
        self._out_encoder_channels = [i['num_chs'] for i in self.feature_info]
        hooks = [dict(module=name, type='forward') for name in self.stage_names]
        self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def forward_features(self, x):
        """Forward method for getting backbone feature maps.
           They are mainly used for segmentation and detection tasks.
        """
        last_features = self(x)
        backbone_features = self.feature_hooks.get_output(x.device)
        backbone_features = list(backbone_features.values())
        backbone_features = [x] + backbone_features
        return last_features, backbone_features

    @property
    def out_encoder_channels(self) -> Tuple[int]:
        """Number of output feature channels - channels after forward_features method."""
        if self._out_encoder_channels is None:
            raise ValueError('TorchOk Backbones must have self._out_feature_channels attribute.')
        return tuple(self._out_encoder_channels)

