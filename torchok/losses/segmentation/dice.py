""" TorchOK DiceLoss module.
Adapted from https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/dice.py
Copyright (c) Eugene Khvedchenya
Licensed under The MIT License [see LICENSE for details]
"""
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchok.constructor import LOSSES

__all__ = ["DiceLoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def soft_dice_score(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 0,
                    eps: float = 1e-7, dims: Union[int, List[int], Tuple[int, ...]] = None) -> torch.Tensor:
    """Compute Dice score.

    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.

    Args:
        y_pred: Prediction tensor.
        y_true: Ground truth tensor.
        smooth: Smooth value.
        eps: Small value to not division by zero.
        dims: Dims for which Dice score will calculate.

    Returns:
        dice_score: Dice score.

    Raises:
        ValueError: If shape y_pred and y_true don't match.
    """
    if y_pred.size() != y_true.size():
        raise ValueError(f"Shapes of y_pred and y_true don't match: {y_pred.size(), y_true.size()}")
    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)
    dice_score = (2.0 * intersection + smooth) / (cardinality.clamp_min(eps) + smooth)
    return dice_score


def to_tensor(x: Union[torch.Tensor, list, tuple], dtype: torch.dtype = None) -> torch.Tensor:
    """Create torch tensor from torch.Tensor, list, tuple types, with specific dtype.

    Args:
        x: Input to convert.
        dtype: Output tensor dtype.

    Returns:
        Tensor with specific dtype.

    Raises:
        ValueError: If x not in torch.Tensor, list, tuple types.
    """
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x

    raise ValueError("Unsupported input type" + str(type(x)))


@LOSSES.register_class
class DiceLoss(nn.Module):
    """Implementation of Dice loss for image segmentation task. It supports binary, multiclass and multilabel cases"""

    def __init__(self, mode: str, classes: List[int] = None, log_loss: bool = False,
                 from_logits: bool = True, smooth: float = 0, eps: float = 1e-7):
        """Init DiceLoss.

        Args:
            mode: Metric mode {'binary', 'multiclass', 'multilabel'}.
            classes: Optional list of classes that contribute in loss computation.
                By default, all channels are included.
            log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`.
            from_logits: If True assumes input is raw logits.
            smooth: Smooth value.
            eps: Small epsilon for numerical stability.

        Raises:
            ValueError: If mode not in set={'binary', 'multiclass', 'multilabel'}.
            ValueError: If classes parameter is not None in binary mode.
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        if mode not in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}:
            raise ValueError(f'DiceLoss initialize. Mode {mode} does not supper. Please choose one of from'
                             f'{[BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE]}.')
        super().__init__()
        self.mode = mode
        if classes is not None:
            if mode == BINARY_MODE:
                raise ValueError('DiceLoss initialize. Masking classes is not supported with mode=binary')
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward method for Dice loss.

        Args:
            input: NxCxHxW if mode is multiclass or multilabel and NxHxW if mode is binary
            target: NxHxW

        Returns:
            loss: Dice loss value - scalar.

        Raises:
            ValueError: If shape input tensor not match with target.
        """
        if self.mode in {BINARY_MODE, MULTILABEL_MODE} and input.shape != target.shape:
            raise ValueError(f"Shapes of input {input.shape} and target {target.shape} tensors don't match!")
        elif self.mode == MULTICLASS_MODE and input[:, 0].shape != target.shape:
            raise ValueError(f"Shapes of input {input.shape} and target {target.shape} tensors don't match!")

        bs = input.shape[0]
        num_classes = input.shape[1] if self.mode in {MULTICLASS_MODE, MULTILABEL_MODE} else 1

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                input = input.log_softmax(dim=1).exp()
            else:
                input = F.logsigmoid(input).exp()

        dims = (0, 2)

        if self.mode == BINARY_MODE:
            target = target.view(bs, 1, -1)
            input = input.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            target = target.view(bs, -1)
            input = input.view(bs, num_classes, -1)

            target = F.one_hot(target, num_classes)  # N,H*W -> N,H*W, C
            target = target.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            target = target.view(bs, num_classes, -1)
            input = input.view(bs, num_classes, -1)

        scores = soft_dice_score(input, target.type_as(input), self.smooth, self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = target.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()
