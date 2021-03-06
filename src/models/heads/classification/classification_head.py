from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

from src.constructor import HEADS
from src.models.base import BaseModel


@HEADS.register_class
class ClassificationHead(BaseModel):
    """Classification head for basic input features."""

    def __init__(self, in_channels: int, num_classes: int, drop_rate: float = 0.0, bias: bool = True):
        """Init ClassificationHead.

        Shape of the input features is (\*, in_features) and shape of the output is (\*, num_classes),
        where * means any number of dimensions. At output the logits are returned, i.e. no softmax is applied.
        When number of classes is equal to 1, the classification task is considered as being a binary classification,
        so the channels dimension is squeezed.

        Args:
            in_channels: number of input features
            num_classes: number of classes
            drop_rate: dropout rate (applied before linear layer)
            bias: whether to use bias in the linear layer
        """
        super().__init__(in_channels, out_channels=num_classes)
        self.drop_rate = drop_rate
        self.fc = nn.Linear(in_channels, num_classes, bias=bias)

    def forward(self, x: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """Forward single input ``x``.

        Args:
            x: Input tensor. 
        """
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        x = self.fc(x)

        if self._out_channels == 1:
            x = x[..., 0]

        return x
