import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchok.constructor import HEADS
from torchok.models.base import BaseModel


@HEADS.register_class
class ArcFaceHead(BaseModel):
    """Implement of arc margin distance. Classification head.

    ArcFace paper: https://arxiv.org/pdf/1801.07698.pdf
    Code: https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """

    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 scale: float = None,
                 margin: float = None,
                 easy_margin: bool = False,
                 dynamic_margin: bool = False,
                 num_warmup_steps: int = None,
                 min_margin: float = None):
        """Init ArcFaceHead class.

        Args:
            in_channels: Size of each input sample.
            num_classes: number of classes.
            scale: Feature scale.
            margin: Angular margin.
            easy_margin: Easy margin.
            dynamic_margin: If True margin will increase from min_margin to margin,
                step by step(num_warmup_steps times).
            num_warmup_steps: Number steps with dynamic margin.
            min_margin: Initial margin in dynamic_margin mode.

        Raises:
            ValueError: if num_warmup_steps or min_margin is None, when `dynamic_margin` is True.
        """
        super().__init__(in_channels, out_channels=num_classes)

        if scale is None:
            p = .999
            c_1 = (num_classes - 1)
            scale = c_1 / num_classes * math.log(c_1 * p / (1 - p)) + 1

        if margin is None:
            if in_channels == 2:
                margin = .9 - math.cos(2 * math.pi / num_classes)
            else:
                margin = .5 * num_classes / (num_classes - 1)

        self.dynamic_margin = dynamic_margin

        if self.dynamic_margin:
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
        self.__cos_m = torch.scalar_tensor(math.cos(self.margin))
        self.__sin_m = torch.scalar_tensor(math.sin(self.margin))
        self.__th = torch.scalar_tensor(math.cos(math.pi - self.margin))
        self.__mm = torch.scalar_tensor(math.sin(math.pi - self.margin) * self.margin)

        self.weight = nn.Parameter(torch.zeros(num_classes, in_channels), requires_grad=True)

        nn.init.xavier_uniform_(self.weight)

    def __update_margin(self) -> None:
        if self.dynamic_margin and self.__step <= self.num_warmup_steps:
            frac = self.__step / self.num_warmup_steps
            margin_gap = self.max_margin - self.min_margin
            self.margin = self.min_margin + frac * margin_gap
            self.__cos_m = torch.scalar_tensor(math.cos(self.margin))
            self.__sin_m = torch.scalar_tensor(math.sin(self.margin))
            self.__th = torch.scalar_tensor(math.cos(math.pi - self.margin))
            self.__mm = torch.scalar_tensor(math.sin(math.pi - self.margin) * self.margin)
            self.__step += 1

    def __add_margin(self, cosine: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = (cosine * self.__cos_m - sine * self.__sin_m).type(cosine.dtype)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.__th, phi, cosine - self.__mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = torch.where(one_hot == 1, phi, cosine)
        output *= self.scale

        return output

    def forward(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward method.

        Args:
            input: Input tensor.
            target: Target tensor. It may be None, if training mode off (inference stage).

        Raises:
            ValueError: If training mode on and target is None.
        """
        if not self.training:
            return F.linear(input, self.weight)
        elif target is None:
            raise ValueError('Target is None in training mode.')

        x = F.normalize(input)
        weight = F.normalize(self.weight)
        cosine = F.linear(x, weight)
        output = self.__add_margin(cosine, target)
        self.__update_margin()

        return output
