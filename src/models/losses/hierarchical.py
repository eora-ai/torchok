import numpy as np
import torch
import torch.nn as nn

from src.registry import LOSSES

__all__ = ['HierarchicalCrossEntropyLoss']


@LOSSES.register_class
class HierarchicalCrossEntropyLoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self, class_graph, num_leaf_classes, reduction='mean'):
        super().__init__()
        if isinstance(class_graph, str):
            class_graph = torch.load(class_graph)
        self.class_graph = class_graph
        self.bce = torch.nn.BCELoss(reduction='sum')
        self.reduction = reduction

        self.mapping = []
        for i in range(num_leaf_classes):
            self.mapping.append(torch.tensor([i]))

        for i, arr in sorted(class_graph.adjacency()):
            if i >= num_leaf_classes:
                arr = np.fromiter(arr, dtype=int)
                arr = sorted(arr[arr < num_leaf_classes])
                self.mapping.append(torch.tensor(arr))

    def forward(self, input, target):
        probs = torch.softmax(input, dim=1)
        losses = []
        for i, tg in enumerate(target):
            loc_probs = [probs[i, self.mapping[tg]].sum()]
            for j in self.class_graph.predecessors(tg.item()):
                loc_probs.append(probs[i, self.mapping[j]].sum())
            loc_probs = torch.clip(torch.stack(loc_probs), min=0, max=1)
            loc_loss = self.bce(loc_probs, torch.ones_like(loc_probs))
            losses.append(loc_loss)
        loss = torch.stack(losses)
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

    def to(self, *args, **kwargs):
        for i, item in enumerate(self.mapping):
            self.mapping[i] = item.to(*args, **kwargs)
        return super(HierarchicalCrossEntropyLoss, self).to(*args, **kwargs)
