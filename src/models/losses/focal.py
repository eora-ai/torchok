import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import LOSSES


@LOSSES.register_class
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, autobalance=False, ignore_index=-100, eps=1e-12,
                 reduction="mean", normalized=False, reduced_threshold=None):
        """
        Focal loss for multi-class problem.
        :param gamma:
        :param alpha:
        :param autobalance: If True, calculate class balancing weights for every batch.
        :param ignore_index: Targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        :param reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`.
                'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        :param normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        :param reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.normalized = normalized
        self.reduced_threshold = reduced_threshold
        self.eps = eps
        self.autobalance = autobalance

    def forward(self, input, target):
        if target.shape == input.shape:
            input = torch.sigmoid(input)

            if self.ignore_index is not None:
                not_ignored = target != self.ignore_index
                input = torch.where(not_ignored, input, torch.zeros_like(input))
                target = torch.where(not_ignored, target, torch.full_like(input, fill_value=0.5))

            logpt = F.binary_cross_entropy(input, target, reduction="none")
            if self.autobalance:
                alpha = self.get_class_balancing(input, target)
                alpha = alpha * target + (1 - alpha) * (1 - target)
            elif self.alpha is not None:
                alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
            else:
                alpha = None
        elif target.shape == input[:, 0].shape:
            logpt = F.cross_entropy(input, target, reduction="none", ignore_index=self.ignore_index)

            if self.autobalance:
                target = torch.where(target == self.ignore_index, torch.zeros_like(target), target)
                alpha = self.get_class_balancing(input, target)[target]
            else:
                alpha = None
        else:
            raise NotImplementedError(f"Shapes of input `{target.shape}` and target `{input.shape}` don't match.")

        loss = self.focal_loss(logpt, alpha)

        return loss

    def focal_loss(self, logpt: torch.Tensor, alpha: torch.Tensor = None) -> torch.Tensor:
        pt = torch.exp(-logpt)

        # compute the loss
        if self.reduced_threshold is None:
            focal_term = (1 - pt).pow(self.gamma)
        else:
            focal_term = ((1.0 - pt) / self.reduced_threshold).pow(self.gamma)
            focal_term = torch.where(pt < self.reduced_threshold, torch.ones_like(focal_term), focal_term)

        loss = focal_term * logpt

        if self.alpha is not None:
            loss = loss * alpha

        if self.normalized:
            loss = loss / (focal_term.sum() + 1e-5)

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "batchwise_mean":
            loss = loss.sum(0)

        return loss

    @staticmethod
    def get_class_balancing(input, target):
        if torch.is_same_size(input, target):
            return 1 - target.mean()
        else:
            class_occurrence = torch.bincount(target.flatten(), minlength=input.shape[1]).float()
            num_of_classes = torch.nonzero(class_occurrence, as_tuple=False).numel()
            weighted_num_of_classes = target.numel() / num_of_classes

            return weighted_num_of_classes / class_occurrence
