import json

import torch
from torch import nn

from src.registry import LOSSES


@LOSSES.register_class
class HierarchicalMultilabelClassificationLoss(nn.Module):
    """
    Loss for hierarchical multilabel classes, so each item might be labeled by multiple hierarchical classes.
    For example, '1.5.7', '13.4.2', '25.6.1'. The loss is calculated from a single vector of probabilities,
    what is usually given in multilabel classification problem
    """
    def __init__(self, classname2levels_path: str, classname2index_path: str, alpha: float = 0.33,
                 reduction: str = 'mean'):
        """
        Args:
            classname2levels_path: path to mapping class names to hierarchy levels. More precisely,
            - keys: class name (different level in the hierarchical classes tree are separated by '.')
            - values: K lists of tuples of two values (min_range, max_range), representing min-max range values of
            corresponding classes falling into their sectors in the original full-length class vector.
            K - depth of the hierarchy, where 1st level represents deepest hierarchy of the classes tree.
            Example: '27.05.08С': [(1495, 1495), (1456, 1690), (1436, 1706)].
            The ranges reserve some sectors in the multilabel vector and the size of the sector gets larger
            while we go upper in the hierarchical classes tree
            classname2index_path: path to mapping class names to indices in the multilabel vector
            alpha: power of weights given for each class range on each level of the hierarchy -
            the bigger alpha the less power
            reduction: type of reduction for loss (see torch.nn.BCELoss for reduction types)
        """
        super().__init__()
        with open(classname2levels_path, 'r') as f:
            self.classname2levels = json.load(f)
        with open(classname2index_path, 'r') as f:
            self.classname2index = json.load(f)
        self.classindex2levels = {self.classname2index[cl_name]: levels
                                  for cl_name, levels in self.classname2levels.items()}

        self.num_classes = len(self.classname2levels)
        self.num_levels = len(list(self.classname2levels.values())[0])

        class_levels = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float32)
        for cl_idx, levels in self.classindex2levels.items():
            for left, right in levels[::-1]:
                weight = (1. / (right - left + 1)) ** alpha
                class_levels[cl_idx, left: right + 1] = weight
        self.register_buffer('class_levels', class_levels)

        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate loss
        Args:
            input: tensor of shape (N, C) representing probabilities of each multilabel class C
            for each item of batch of size N
            target: tensor of shape (N, C) representing multihot vector for each item of batch of size N -
            1 if i-th class is present, 0 - otherwise, where i is in range [0, C - 1]

        Returns: loss value

        """
        batch_tgt_levels = []

        for t in target:
            tgt_class_indices = torch.nonzero(t, as_tuple=True)[0]
            tgt_levels = self.class_levels[tgt_class_indices]
            tgt_levels = tgt_levels.max(0)[0]
            batch_tgt_levels.append(tgt_levels)

        batch_tgt_levels = torch.stack(batch_tgt_levels)
        loss = self.bce(input, batch_tgt_levels)

        return loss
