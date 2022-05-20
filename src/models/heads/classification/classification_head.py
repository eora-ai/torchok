import torch.nn.functional as F
from torch import nn, Tensor

from src.constructor import HEADS


class AbstractHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        raise NotImplementedError()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

@HEADS.register_class
class ClassificationHead(AbstractHead):
    def __init__(self, in_features, num_classes, drop_rate=0.0, bias=True):
        super().__init__(in_features, num_classes)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.fc = nn.Linear(in_features, num_classes, bias=bias)
        self.init_weights()

    def forward(self, x: Tensor, targets: Tensor = None) -> Tensor:
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        if self.num_classes == 1:
            x = x[:, 0]

        return x
