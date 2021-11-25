import torch.nn as nn

from .utils.registry import register_model

__all__ = ['IdentityBackbone']


@register_model
class IdentityBackbone(nn.Module):
    def __init__(self, num_features, pretrained=None, in_chans=None):
        super().__init__()
        self.num_features = num_features
        self.identity = nn.Identity()

    def forward(self, x):
        y = self.identity(x)
        return y
