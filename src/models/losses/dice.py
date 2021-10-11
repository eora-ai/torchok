from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.registry import LOSSES

# Copypasted from https://github.com/BloodAxe/pytorch-toolbelt

__all__ = ["DiceLoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"
MULTILABEL_MODE = "multilabel"


def soft_dice_score(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=0, eps=1e-7, dims=None) -> torch.Tensor:
    """
    :param y_pred:
    :param y_true:
    :param smooth:
    :param eps:
    :return:
    Shape:
        - Input: :math:`(N, NC, *)` where :math:`*` means any number
            of additional dimensions
        - Target: :math:`(N, NC, *)`, same shape as the input
        - Output: scalar.
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


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
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
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(self, mode: str, classes: List[int] = None, log_loss=False, from_logits=True, smooth=0, eps=1e-7):
        """
        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param eps: Small epsilon for numerical stability
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: NxCxHxW if mode is multiclass or multilabel and NxHxW if mode is binary
        :param target: NxHxW
        :return: scalar
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
