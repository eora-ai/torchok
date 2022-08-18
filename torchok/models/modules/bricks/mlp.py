from typing import Optional

import torch.nn as nn
from torch import Tensor


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: nn.Module = nn.GELU):
        """Init Mlp.

        Args:
            in_features: Input features.
            hidden_features: Hidden features.
            out_features: Output features.
            act_layer: Activation layer.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
