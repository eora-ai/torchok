from torch import nn, Tensor
import torch.nn.functional as F

from torchok.constructor import HEADS
from torchok.models.heads.base import AbstractHead


@HEADS.register_class
class ClassificationHead(AbstractHead):
    """Classification head for basic input features."""

    def __init__(self, in_features: int, num_classes: int, drop_rate: float = 0.0, bias: bool = True):
        """Init ClassificationHead.

        Shape of the input features is (\*, in_features) and shape of the output is (\*, num_classes),
        where * means any number of dimensions. At output the logits are returned, i.e. no softmax is applied.
        When number of classes is equal to 1, the classification task is considered as being a binary classification,
        so the channels dimension is squeezed.

        Args:
            in_features: number of input features
            num_classes: number of classes
            drop_rate: dropout rate (applied before linear layer)
            bias: whether to use bias in the linear layer
        """
        super().__init__(in_features, num_classes)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward single input ``x``.
        
        Args:
            x: input features of shape (\*, in_features), where \* means any number of dimensions,
            in_features - number of features configured for the head
        """
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        
        x = self.fc(x)

        if self.num_classes == 1:
            x = x[..., 0]

        return x
