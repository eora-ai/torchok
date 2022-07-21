""" TorchOK DropPath module.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py
Copyright 2020 Ross Wightman
Licensed under The Apache 2.0 License [see LICENSE for details]
"""
import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float = 0., 
              training: bool = False, scale_by_keep: bool = True) -> torch.Tensor:
    # TODO check that DaVit does not add one more implementation
    """Drop paths (Stochastic Depth).
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956.
    This implementation multiply input tensor to tensor with Bernully distribution, and then
    divide to keep_prob if scale+by_keep is True. 

    Args:
        x: Input tensor.
        drop_prob: Drop path probability.
        training: If use drop_path.
        scale_by_keep: If scale output tensor by 1/drop_prob value.

    Returns:
        x: Drop path output.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        """Init DropPath.
        
        Args:
            drop_prob: Drop path probability.
            scale_by_keep: If scale output tensor by 1/drop_prob value.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
