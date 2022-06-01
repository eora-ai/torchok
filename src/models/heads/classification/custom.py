from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constructor import HEADS
from src.models.heads.base import AbstractHead


@HEADS.register_class
class ClassificationHead(AbstractHead):
    """Implement of arc margin distance. Classification head.

    ArcFace paper: https://arxiv.org/pdf/1801.07698.pdf
    Code: https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """

    def __init__(self,
                 in_features: int,
                 out_features: int):
        """Init ArcFaceHead class.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.

        Raises:
            ValueError: if num_warmup_steps or min_margin is None, when `dynamic_margin` is True.
        """
        super().__init__(in_features, out_features)
        
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input, target: torch.Tensor = None) -> torch.Tensor:
        return self.linear(input)
