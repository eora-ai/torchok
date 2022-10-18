from abc import ABC
from typing import List, Tuple

from timm.models.features import FeatureHooks
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.batchnorm import _BatchNorm

from torchok.models.base import BaseModel


def set_batchnorm_eval(m):
    """
    Credits to Filip Radenovic (https://github.com/filipradenovic/cnnimageretrieval-pytorch)
    """

    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.track_running_stats = False


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
        last_features = self(x)  # noqa
        backbone_features = self.feature_hooks.get_output(x.device)
        backbone_features = list(backbone_features.values())
        return [x] + backbone_features

    @property
    def out_encoder_channels(self) -> Tuple[int]:
        """Number of output feature channels - channels after forward_features method."""
        if self._out_encoder_channels is None:
            raise ValueError('TorchOk Backbones must have self._out_feature_channels attribute.')
        return tuple(self._out_encoder_channels)

    def _freeze_stages(self):
        if self.bn_requires_grad:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    for p in m.parameters():
                        p.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer frozen."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            self.apply(set_batchnorm_eval)


class BackboneWrapper(Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone.forward_features(x)

    @property
    def out_encoder_channels(self):
        return self.backbone.out_encoder_channels
