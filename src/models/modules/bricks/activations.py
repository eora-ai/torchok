import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class HSigmoid(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out
