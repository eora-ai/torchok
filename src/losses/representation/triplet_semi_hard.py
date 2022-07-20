# Copyright from https://github.com/alfonmedela/triplet-loss-pytorch. All Rights Reserved.
# Apache-2.0 License

from typing import List
import torch
import torch.nn as nn
from src.constructor import LOSSES
import torch.nn.functional as F


def pairwise_distance_torch(embeddings):
    """Computes the pairwise distance matrix with numerical stability.

    Output[i, j] = || embeddings[i, :] - embeddings[j, :] ||_2
       
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].

    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    device = embeddings.device
    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.], device=device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(
        torch.ones(pairwise_distances.shape[0]))
    mask_offdiagonals = mask_offdiagonals.to(device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def compute_triplet_semi_hard_loss(y_true, y_pred, margin=1.0) -> torch.Tensor:
    """Computes the triplet loss_functions with semi-hard negative mining.
       
    Args:
        y_true: 1-D integer Tensor with shape [batch_size] of multi-class integer labels.
        y_pred: 2-D float Tensor of l2 normalized embedding vectors.
        margin: Float, margin term in the loss_functions definition. Default value is 1.0.

    Returns: 
        triplet_loss: Triplet semi hard loss.
    """
    device = y_pred.device
    labels, embeddings = y_true, y_pred

    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)

    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    greater = pdist_matrix_tile > transpose_reshape

    mask = adjacency_not_tile & greater

    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + \
                      axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + \
                      axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size, device=device))
    num_positives = mask_positives.sum()

    if num_positives == 0:
        num_positives += 1e-5

    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.], device=device))).sum() 
    triplet_loss = triplet_loss / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)

    return triplet_loss


@LOSSES.register_class
class TripletSemiHardLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletSemiHardLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        return compute_triplet_semi_hard_loss(y_true=target, y_pred=input, margin=self.margin)
