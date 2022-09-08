import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from torchok.constructor import LOSSES


@LOSSES.register_class
class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss with ability to load pos_weights from json file (dict) or config (list)."""
    def __init__(self, weight: torch.Tensor = None, reduction: str = 'mean', pos_weight: Union[str, list] = None):
        """BCEWithLogitsLossX init.

        Args:
            weight: A manual rescaling weight given to the loss of each batch element. If given,
                has to be a Tensor of size nbatch.
            reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction
                will be applied, 'mean': the sum of the output will be divided by the number of elements in the output,
                'sum': the output will be summed.
            pos_weight: A weight of positive examples. Must be a vector with length equal to the number of classes.
                Can be string - json file with keys - class index and value - weight, or can be list - readable from
                yaml config with the weights.
        """
        if pos_weight is not None:
            if isinstance(pos_weight, str):
                pos_weight_path = pos_weight
                with open(pos_weight_path) as weights_file:
                    weights_dict = json.load(weights_file)

                num_classes = len(weights_dict)
                pos_weight = torch.ones([num_classes])
                for k, v in weights_dict.items():
                    pos_weight[int(k)] = v
                logging.info(f'using pos_weights loaded from {pos_weight_path}')
            elif isinstance(pos_weight, list):
                pos_weight = torch.tensor(pos_weight, dtype=torch.float)
                logging.info('using pos_weights loaded from cfg file')
        super().__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target.float(),
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)
