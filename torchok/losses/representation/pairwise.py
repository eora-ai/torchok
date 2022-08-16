from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn import Module
from torchok.constructor import LOSSES


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine similarity matrix between embeddings matrices x and y
    Args:
        x: First embeddings matrix, shape (N, D), where N - number of embeddings, D - embeddings dimension,
            dtype=float32
        y: Second embeddings matrix, shape (M, D), where M - number of embeddings, D - embeddings dimension,
            dtype=float32

    Returns:
        Similarity matrix of shape (N, M), dtype=float32

    """
    # (N, D), (D, M) -> (M, M)
    x, y = x.float(), y.float()
    distance_matrix = torch.matmul(x, y.transpose(1, 0))

    return distance_matrix


def euclidean_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes euclidean similarity matrix between embeddings matrices x and y
    Args:
        x: First embeddings matrix, shape (N, D), where N - number of embeddings, D - embeddings dimension,
            dtype=float32
        y: Second embeddings matrix, shape (M, D), where M - number of embeddings, D - embeddings dimension,
            dtype=float32

    Returns:
        Similarity matrix of shape (N, M), dtype=float32

    """
    # ||x - y|| = sum(x^2) + sum(y^2) - 2*x*y
    x, y = x.float(), y.float()
    x_norm = x.pow(2).sum(1)
    y_norm = y.pow(2).sum(1)
    distance_matrix = x_norm.view(-1, 1) + y_norm.view(1, -1) - 2 * torch.matmul(x, y.transpose(1, 0))
    distance_matrix = torch.sqrt(F.relu(distance_matrix))

    return distance_matrix


class BasePairwiseLoss(Module):
    """
    Contains basic operations on loss: regularization and reduction
    Args:
        margin: margin that controls how far samples take from easy decision boundary
        reg: type of regularization that is applied to input embeddings, possible values: L1, L2.
            If None, no regularization is applied
        reduction: type of reduction for output loss vector, possible values: mean, sum.
            If None, no reduction is applied
    """
    def __init__(self, reg: Optional[str] = None, reduction: Optional[str] = 'mean'):
        super().__init__()
        self.reg = reg
        self.reduction = reduction
        self.eps = 1e-5

    def regularize(self, L: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Adds regularization factor to the given loss
        Args:
            L: Current loss value, shape (B,) where B - batch size
            emb: Embeddings tensor, shape (B, D) where B - batch size, D - embeddings dimension

        Returns:
            Updated loss value, shape (B), where B - batch size

        """
        if self.reg is not None and self.reg == 'L2':
            return L + 0.001 * torch.norm(emb, p=None, dim=1)
        elif self.reg is not None and self.reg == 'L1':
            return L + 0.001 * emb.abs().sum(1)
        elif self.reg is not None:
            raise ValueError(f'Unknown regularization type: {self.reg}')
        else:
            return L

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
        elif self.reg is not None:
            raise ValueError(f'Unknown reduction type: {self.reduction}')

        return L


class GeneralPairWeightingLoss(BasePairwiseLoss):
    """
    General Pair Weighting framework as described in paper `Cross-Batch Memory for Embedding Learning`_
    Args:
        margin: margin that controls how far samples take from easy decision boundary
        reg: type of regularization that is applied to input embeddings, possible values: L1, L2.
            If None, no regularization is applied
        reduction: type of reduction for output loss vector, possible values: mean, sum.
            If None, no reduction is applied

    .. _Cross-Batch Memory for Embedding Learning:
            https://arxiv.org/abs/1912.06798
    """
    def __init__(self, margin: float, reg: Optional[str] = None, reduction: Optional[str] = 'mean'):
        super().__init__(reg=reg, reduction=reduction)
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
        S = euclidean_similarity(emb1, emb2)    # range [0, 2]
        mu = self.margin
        L = (1. - R) * F.relu(mu - S).pow(2) + R * S.pow(2)
        L = L.sum(1)

        return L
