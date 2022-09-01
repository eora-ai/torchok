from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn import Module
from torchok.constructor import LOSSES


class BasePairwiseLoss(Module):
    """Contains basic operations on loss: regularization and reduction."""

    def __init__(self, reg: Optional[str] = None, reduction: Optional[str] = 'mean', eps: Optional[float] = 1e-3):
        """Init BasePairwiseLoss.

        Args:
            margin: Margin that controls how far samples take from easy decision boundary.
            reg: Type of regularization that is applied to input embeddings, possible values: L1, L2.
                If None, no regularization is applied
            reduction: Type of reduction for output loss vector, possible values: mean, sum.
                If None, no reduction is applied
            eps: Eps (default: 1e-3).
        """
        super().__init__()
        self.reg = reg
        self.reduction = reduction
        self.eps = eps

    def regularize(self, L: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Adds regularization factor to the given loss
        Args:
            L: Current loss value, shape (B,) where B - batch size
            emb: Embeddings tensor, shape (B, D) where B - batch size, D - embeddings dimension

        Returns:
            Updated loss value, shape (B), where B - batch size

        """
        if self.reg is None:
            return L
        elif self.reg == 'L1':
            return L + self.eps * emb.abs().sum(1)
        elif self.reg == 'L2':
            return L + self.eps * torch.norm(emb, p=None, dim=1)
        else:
            raise ValueError(f'Unknown regularization type: {self.reg}')

    def apply_reduction(self, L: torch.Tensor) -> torch.Tensor:
        """
        Reduces loss by batch dimension to a scalar if reduction is specified
        Args:
            L: Current loss value, shape (B,) where B - batch size

        Returns:
            Updated loss value, shape (B), where B = 1 if reduction is specified or B - batch size otherwise

        """
        if self.reduction == 'mean':
            L = L.mean()
        elif self.reduction == 'sum':
            L = L.sum()
        else:
            raise ValueError(f'Unknown reduction type: {self.reduction}')

        return L


class GeneralPairWeightingLoss(BasePairwiseLoss):
    """General Pair Weighting framework as described in paper `Cross-Batch Memory for Embedding Learning`_

    .. _Cross-Batch Memory for Embedding Learning:
            https://arxiv.org/abs/1912.06798
    """
    def __init__(self, margin: float, reg: Optional[str] = None,
                 reduction: Optional[str] = 'mean', eps: Optional[float] = 1e-3):
        """Init GeneralPairWeightingLoss.

        Args:
            margin: Margin that controls how far samples take from easy decision boundary.
            reg: Type of regularization that is applied to input embeddings, possible values: L1, L2.
                If None, no regularization is applied
            reduction: Type of reduction for output loss vector, possible values: mean, sum.
                If None, no reduction is applied
            eps: Eps (default: 1e-3).
        """
        super().__init__(reg=reg, reduction=reduction, eps=eps)
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        GPW losses can be calculated on embeddings tensor from the current batch only or the current batch and memory.
            In the first case parameters emb1 and emb2 should be the same tensors
        Args:
            emb1: First embeddings tensor, shape (B, D) where B - batch size, D - embeddings dimension, dtype=float32
            emb2: Second embeddings tensor, shape (M, D) where M - memory size, D - embeddings dimension, dtype=float32
            R: Relevance matrix, where values 1 mean the samples are similar to each other, 0 - otherwise,
                shape (B, M), where B - batch size, M - memory/batch size, dtype=float32

        Returns:
            Loss value, shape (B,) where B = 1 if reduction is specified or B - batch size otherwise

        """
        L = self.calc_loss(emb1, emb2, R)
        L = self.regularize(L, emb1)
        L = self.apply_reduction(L)

        return L

    def calc_loss(self, emb1: torch.Tensor, emb2: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Loss value, shape (B,) where B - batch size

        See documentation of `forward` method

        """
        raise NotImplementedError()


@LOSSES.register_class
class ContrastiveLoss(GeneralPairWeightingLoss):
    """
    Contrastive loss
    See base class documentation for more details
    """
    def calc_loss(self, emb1: torch.Tensor, emb2: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        See documentation of base `forward` method

        """
        S = torch.cdist(emb1, emb2, p=2)
        mu = self.margin
        L = (1. - R) * F.relu(mu - S).pow(2) + R * S.pow(2)
        L = L.sum(1)

        return L
