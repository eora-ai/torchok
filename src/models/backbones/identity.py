import torch.nn as nn

from .base import BackboneBase
from .utils.registry import register_model

__all__ = ['IdentityBackbone']


@register_model
class IdentityBackbone(BackboneBase):
    def __init__(self, num_features, pretrained=None, in_chans=None):
        super().__init__()
        self.num_features = num_features
        self.feature_info = []

    def forward_features(self, x):
        return x
