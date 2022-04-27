import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constructor import HEADS, CLASSIFICATION_HEADS

#from src.models.heads.base import AbstractHead
from abc import ABC, abstractmethod



class AbstractHead(nn.Module, ABC):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

@HEADS.register_class
@CLASSIFICATION_HEADS.register_class
class ArcFaceHead(AbstractHead):
    """Implement of large arc margin distance."""

    def __init__(self, in_features, num_classes, scale=None, margin=None, easy_margin=False,
                 dynamic_margin=False, num_warmup_steps=None, min_margin=None):

        """Init ArcFaceHead class.

        Args:
            in_features: size of each input sample
            num_classes: size of each output sample
            scale: norm of output features
            margin: margin for the positive class

        Raises:
            ValueError: sometext
        """
        super().__init__(in_features, num_classes)
        self.num_classes = num_classes
        if scale is None:
            p = .999
            c_1 = (num_classes - 1)
            scale = c_1 / num_classes * math.log(c_1 * p / (1 - p)) + 1

        if margin is None:
            if in_features == 2:
                margin = .9 - math.cos(2 * math.pi / num_classes)
            else:
                margin = .5 * num_classes / (num_classes - 1)

        self.dynamic_margin = dynamic_margin
        if dynamic_margin:
            if num_warmup_steps is None or not isinstance(num_warmup_steps, int):
                raise ValueError('`num_warmup_steps` must be positive int when `dynamic_margin` is True')
            if min_margin is None:
                raise ValueError('`min_margin` must be float when `dynamic_margin` is True')
            self.num_warmup_steps = num_warmup_steps
            self.min_margin = min_margin
            self.max_margin = margin
            self.margin = min_margin
            self.register_buffer('step', torch.tensor(0))
        else:
            self.margin = margin

        self.scale = scale
        self.easy_margin = easy_margin
        self.cos_m = torch.scalar_tensor(math.cos(self.margin))
        self.sin_m = torch.scalar_tensor(math.sin(self.margin))
        self.th = torch.scalar_tensor(math.cos(math.pi - self.margin))
        self.mm = torch.scalar_tensor(math.sin(math.pi - self.margin) * self.margin)

        self.weight = nn.Parameter(torch.zeros(num_classes, in_features),
                                   requires_grad=True)
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

    def _update_margin(self):
        if self.dynamic_margin and self.step <= self.num_warmup_steps:
            frac = self.step / self.num_warmup_steps
            self.margin = self.min_margin * (1 - frac) + self.max_margin * frac
            self.cos_m = torch.scalar_tensor(math.cos(self.margin))
            self.sin_m = torch.scalar_tensor(math.sin(self.margin))
            self.th = torch.scalar_tensor(math.cos(math.pi - self.margin))
            self.mm = torch.scalar_tensor(math.sin(math.pi - self.margin) * self.margin)
            self.step += 1

    def _add_margin(self, cosine, target=None):
        if target is None:
            return cosine

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = (cosine * self.cos_m - sine * self.sin_m).type(cosine.dtype)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # ---------------------- convert label to one-hot ---------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = torch.where(one_hot == 1, phi, cosine)
        output *= self.scale
        self._update_margin()
        return output

    def forward(self, input: torch.Tensor, target: torch.Tensor = None):
        # ---------------------- cos(theta) & phi(theta) ----------------------
        if not self.training:
            return F.linear(input, self.weight)

        x = F.normalize(input)
        weight = F.normalize(self.weight)
        cosine = F.linear(x, weight)
        return self._add_margin(cosine, target)
