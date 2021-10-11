from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn import Module
from src.registry import LOSSES


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


@LOSSES.register_class
class TripletLoss(GeneralPairWeightingLoss):
    """
    Triplet loss
    Args:
        mode: Mode in which to calculate loss. Possible values:
            - weighted: loss is calculated as described in paper `Cross-Batch Memory for Embedding Learning`_
            - hard: loss is calculated as provided in the implementation of `XBM retrieval benchmark`_

    .. _Cross-Batch Memory for Embedding Learning:
    https://arxiv.org/abs/1912.06798
    .. _XBM retrieval benchmark:
    https://github.com/MalongTech/research-xbm/tree/master/ret_benchmark
    See base class documentation for more details
    """
    def __init__(self, margin: float, reg: Optional[str] = None, reduction: Optional[str] = 'mean', mode: str = 'hard'):
        super().__init__(margin, reg, reduction)

        if mode == 'weighted':
            self.loss_func = self._compute_weighted_loss
        elif mode == 'hard':
            self.loss_func = self._compute_hard_loss
        else:
            raise ValueError(f'Unknown loss type: {mode}')

    def calc_loss(self, emb1: torch.Tensor, emb2: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        See documentation of base `forward` method

        """
        S = cosine_similarity(emb1, emb2)  # range [0, 1]
        L = self.loss_func(S, R)

        return L

    def _compute_weighted_loss(self, S: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Computes weighted loss as described in paper `Cross-Batch Memory for Embedding Learning`_
        Args:
            S: Cosine similarity matrix, values in range [0, 1] where higher value means samples are more similar
                to each other, shape (B, M), where B - batch size, M - memory/batch size, dtype=float32
            R: Relevance matrix, where values 1 mean the samples are similar to each other, 0 - otherwise,
                shape (B, M), where B - batch size, M - memory/batch size, dtype=float32

        Returns:
            Loss value, shape (B,) where B - batch size

        .. _Cross-Batch Memory for Embedding Learning:
            https://arxiv.org/abs/1912.06798
        """
        mu = self.margin
        batch_size = R.shape[0]
        memory_size = R.shape[0]
        P = torch.empty_like(S)
        N = torch.empty_like(S)

        for i in range(batch_size):
            for j in range(memory_size):
                p = (R * (S > S[i, j] - mu))[i].sum()
                n = ((1 - R) * (S < S[i, j] + mu))[i].sum()

                P[i, j] = p
                N[i, j] = n

        L = R * N * S + F.relu((1 - R) * P * (1. - S))  # positive pairs - negative pairs
        L = L.sum(1)

        return L

    def _compute_hard_loss(self, S: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Computes hard loss as provided in the implementation of `XBM retrieval benchmark`_
        Args:
            S: Cosine similarity matrix, values in range [0, 1] where higher value means samples are more similar
                to each other, shape (B, M), where B - batch size, M - memory/batch size, dtype=float32
            R: Relevance matrix, where values 1 mean the samples are similar to each other, 0 - otherwise,
                shape (B, M), where B - batch size, M - memory/batch size, dtype=float32

        Returns:
            Loss value, shape (B,) where B - batch size

        .. _XBM retrieval benchmark:
            https://github.com/MalongTech/research-xbm/tree/master/ret_benchmark
        """
        n = S.size(0)
        # Compute similarity matrix
        sim_mat = S
        # split the positive and negative pairs
        eyes_ = torch.eye(n, dtype=torch.uint8).cuda()
        pos_mask = R
        neg_mask = 1 - pos_mask
        pos_mask[:, :n] = pos_mask[:, :n] - eyes_

        loss = list()
        neg_count = list()
        for i in range(n):
            pos_pair_idx = torch.nonzero(pos_mask[i, :]).view(-1)
            if pos_pair_idx.shape[0] > 0:
                pos_pair_ = sim_mat[i, pos_pair_idx]
                pos_pair_ = torch.sort(pos_pair_)[0]

                neg_pair_idx = torch.nonzero(neg_mask[i, :]).view(-1)
                neg_pair_ = sim_mat[i, neg_pair_idx]
                neg_pair_ = torch.sort(neg_pair_)[0]

                select_pos_pair_idx = torch.nonzero(
                    pos_pair_ < neg_pair_[-1] + self.margin
                ).view(-1)
                pos_pair = pos_pair_[select_pos_pair_idx]

                select_neg_pair_idx = torch.nonzero(
                    neg_pair_ > max(0.6, pos_pair_[-1]) - self.margin
                ).view(-1)
                neg_pair = neg_pair_[select_neg_pair_idx]

                pos_loss = torch.sum(1 - pos_pair)
                if len(neg_pair) >= 1:
                    neg_loss = torch.sum(neg_pair)
                    neg_count.append(len(neg_pair))
                else:
                    neg_loss = 0
                loss.append(pos_loss + neg_loss)
            else:
                loss.append(0)

        loss = sum(loss) / n
        # Hack to overcome case when all items were 0 scalars
        if isinstance(loss, float):
            loss = torch.tensor(loss, requires_grad=True, device=S.device)

        return loss


@LOSSES.register_class
class MultiSimilarityLoss(GeneralPairWeightingLoss):
    """
    Multi-similarity loss as described in paper `Cross-Batch Memory for Embedding Learning`_
    Args:
        margin: Margin in this loss differs from what usually understood. Here this is a threshold value used for
            calculating exponential weight
        scale_pos: Scaling factor for distance between positives and margin
        scale_neg: Scaling factor for distance between negatives and margin

    .. _Cross-Batch Memory for Embedding Learning:
        https://arxiv.org/abs/1912.06798

    """
    def __init__(self, margin: float, scale_pos: float, scale_neg: float, reg: Optional[str] = None,
                 reduction: Optional[str] = 'mean'):
        super().__init__(margin, reg, reduction)
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def calc_loss(self, emb1: torch.Tensor, emb2: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        See documentation of base `forward` method

        """
        S = cosine_similarity(emb1, emb2)  # range [0, 1]

        n = S.size(0)
        # Compute similarity matrix
        sim_mat = S

        epsilon = 1e-5
        loss = list()
        neg_count = 0
        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], R[i] == 1)
            pos_pair = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair = torch.masked_select(sim_mat[i], R[i] == 0)

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            neg_count += len(neg_pair)

            # weighting step
            pos_loss = (
                    1.0
                    / self.scale_pos
                    * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.margin)))
            ))
            neg_loss = (
                    1.0
                    / self.scale_neg
                    * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.margin)))
            ))
            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        # Hack to overcome case when all items were 0 scalars
        if isinstance(loss, float):
            loss = torch.tensor(loss, requires_grad=True, device=S.device)

        return loss
