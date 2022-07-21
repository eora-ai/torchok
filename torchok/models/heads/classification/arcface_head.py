from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchok.constructor import HEADS
from torchok.models.heads.base import AbstractHead


@HEADS.register_class
class ArcFaceHead(AbstractHead):
    """Implement of arc margin distance. Classification head.

    ArcFace paper: https://arxiv.org/pdf/1801.07698.pdf
    Code: https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int,
                 scale: float = 30.0,
                 margin: float = 0.5,
                 easy_margin: bool = False,
                 dynamic_margin: bool = False,
                 num_warmup_steps: int = None,
                 min_margin: float = None):
        """Init ArcFaceHead class.

        Args:
            in_features: Size of each input sample.
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
        super().__init__(in_features, num_classes)

        self.__num_classes = num_classes

        if scale is None:
            p = .999
            c_1 = (num_classes - 1)
            scale = c_1 / num_classes * math.log(c_1 * p / (1 - p)) + 1

        if margin is None:
            if in_features == 2:
                margin = .9 - math.cos(2 * math.pi / num_classes)
            else:
                margin = .5 * num_classes / (num_classes - 1)

        self.__dynamic_margin = dynamic_margin

        if self.__dynamic_margin:
            if num_warmup_steps is None or not isinstance(num_warmup_steps, int):
                raise ValueError('`num_warmup_steps` must be positive int when `dynamic_margin` is True')
            if min_margin is None:
                raise ValueError('`min_margin` must be float when `dynamic_margin` is True')
            self.__num_warmup_steps = num_warmup_steps
            self.__min_margin = min_margin
            self.__max_margin = margin
            self.__margin = min_margin
            self.__register_buffer('step', torch.tensor(0))
        else:
            self.__margin = margin

        self.__scale = scale
        self.__easy_margin = easy_margin
        self.__cos_m = torch.scalar_tensor(math.cos(self.__margin))
        self.__sin_m = torch.scalar_tensor(math.sin(self.__margin))
        self.__th = torch.scalar_tensor(math.cos(math.pi - self.__margin))
        self.__mm = torch.scalar_tensor(math.sin(math.pi - self.__margin) * self.__margin)

        self.__weight = nn.Parameter(torch.zeros(num_classes, in_features), requires_grad=True)

        nn.init.xavier_uniform_(self.__weight)

    def __update_margin(self) -> None:
        if self.__dynamic_margin and self.__step <= self.__num_warmup_steps:
            frac = self.__step / self.__num_warmup_steps
            margin_gap = self.__max_margin - self.__min_margin
            self.__margin = self.__min_margin + frac * margin_gap
            self.__cos_m = torch.scalar_tensor(math.cos(self.__margin))
            self.__sin_m = torch.scalar_tensor(math.sin(self.__margin))
            self.__th = torch.scalar_tensor(math.cos(math.pi - self.__margin))
            self.__mm = torch.scalar_tensor(math.sin(math.pi - self.__margin) * self.__margin)
            self.__step += 1

    def __add_margin(self, cosine: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = (cosine * self.__cos_m - sine * self.__sin_m).type(cosine.dtype)
        if self.__easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.__th, phi, cosine - self.__mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        output = torch.where(one_hot == 1, phi, cosine)
        output *= self.__scale

        return output

    def forward(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward method.

        Args:
            input: Input tensor.
            target: Target tensor. May be None, if training mode off (inference stage).

        Raises:
            ValueError: If training mode on and target is None.
        """
        if not self.training:
            return F.linear(input, self.__weight)
        elif target is None:
            raise ValueError('Target is None in training mode.')

        x = F.normalize(input)
        weight = F.normalize(self.__weight)
        cosine = F.linear(x, weight)
        output = self.__add_margin(cosine, target)
        self.__update_margin()

        return output

    @property
    def num_classes(self) -> int:
        """Output features or number classes."""
        return self.__num_classes

    @property
    def margin(self) -> float:
        """Angular margin."""
        return self.__margin

    @property
    def scale(self) -> float:
        """Feature scale."""
        return self.__scale

    @property
    def dynamic_margin(self) -> bool:
        """Dynamic margin mode."""
        return self.__dynamic_margin

    @property
    def num_warmup_steps(self) -> Optional[int]:
        """It's number of warm-up steps."""
        return self.__num_warmup_steps

    @property
    def min_margin(self) -> Optional[float]:
        """It's initial margin in `dynamic_margin` mode."""
        return self.__min_margin

    @property
    def weights(self) -> torch.Tensor:
        """It's ArcFaceHead weights."""
        return self.__weight
