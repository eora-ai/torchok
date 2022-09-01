from typing import Any, Dict, List, Optional, Tuple

from torch import Tensor
from torch.nn import Module, ModuleList


class JointLoss(Module):
    """Represents wrapper for loss modules.

    This joined loss can be forwarded as a weighted sum of losses and can provide direct access to each loss.
    """

    def __init__(self, losses: List[Module], mappings: List[Dict[str, str]],
                 tags: List[Optional[str]], weights: List[Optional[float]],
                 normalize_weights: bool = True):
        """Init JointLoss.

        Reduction isn't applied, so the included loss modules are responsible for reduction

        Args:
            losses: List of loss modules with their `forward` methods implemented
            mappings: List of mappings, each making relationships from neural network outputs to loss inputs.
            Useful in `Task`s in their `forward_with_gt` methods to support direct passing of tensors
            between a neural network and loss functions
            tags: List of tags applied to each loss from `losses`. Useful when user wants to access individual loss
            by its tag via `JointLoss.__getitem__`. Losses for which tags are `None`, can't be accessed directly and
            `JointLoss.__getitem__` will throw a `ValueError`
            weights: List of scalar weights for loss functions, so that resulting loss value will be:
            `loss = loss_i * weight_i for i in [0, len(losses) - 1]`. The weights must be
            either specified for each loss function or not specified for any loss functions.
            In the last case the weights will be automatically set as equal
            normalize_weights: Either to normalize the weights, so that they sum up to 1 before loss value calculation.

        Raises:
            - ValueError: When only a few weights are specified but not for all the loss modules
        """
        super().__init__()
        self.losses = ModuleList(losses)
        self.tag2loss = {tag: loss for tag, loss in zip(tags, self.losses) if tag is not None}
        self.tags = tags
        self.mappings = mappings

        num_specified_weights = len(list(filter(lambda w: w is not None, weights)))
        if num_specified_weights > 0 and num_specified_weights != len(losses):
            raise ValueError('Loss weights must be either specified for each loss function or '
                             'not specified for any loss function')

        if num_specified_weights == 0:
            self.weights = [1.] * len(self.losses)
        else:
            self.weights = weights

        if normalize_weights:
            self.weights = [w / sum(self.weights) for w in self.weights]

    def forward(self, **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Forward the joined loss module.

        First, individual loss modules are calculated. Then the values are summed up in a weighted average manner
        to form the total loss value.

        Args:
            **kwargs: Any tensors which are supported by the corresponding loss modules.
            The mapping of each loss module is used to map neural network outputs to loss inputs

        Returns:
            Tuple:
            - Tensor, representing total loss value as a weighted sum of all calculated loss values
            - Dict, where key - tag of individual loss module, value - Tensor, representing the loss value.
            Individual values will be return for tagged values only

        Raises:
            - ValueError: When some keys from a mapping cannot be found among neural network outputs
        """
        total_loss = 0.
        tagged_loss_values = {}
        for loss_module, mapping, tag, weight in zip(self.losses, self.mappings, self.tags, self.weights):
            targeted_kwargs = self._parse_match_csv(mapping, **kwargs)
            loss = loss_module(**targeted_kwargs)
            # TODO: condition key
            total_loss = total_loss + loss * weight
            if tag is not None:
                tagged_loss_values[tag] = loss

        return total_loss, tagged_loss_values

    def __getitem__(self, tag: str) -> Module:
        """Provide direct access to loss module by its tag.

        Args:
            tag: Tag of the desired loss module

        Returns: Loss module

        Raises:
            - KeyError: When a specified tag cannot be found (often happens when tags are specified not for each loss)
        """
        if tag in self.tag2loss:
            return self.tag2loss[tag]
        else:
            raise KeyError(f'Cannot access loss {tag}. You should tag your losses for direct access with a tag key')

    @staticmethod
    def _parse_match_csv(mapping: Dict[str, str], **model_outputs) -> Dict[str, Any]:
        targeted_kwargs = {}
        for target_arg, source_arg in mapping.items():
            if source_arg in model_outputs:
                targeted_kwargs[target_arg] = model_outputs[source_arg]
            else:
                raise ValueError(f'Cannot find {source_arg} for your mapping {target_arg} : {source_arg}. '
                                 f'You should either add {source_arg} output to your model or remove the mapping '
                                 f'from configuration')
        return targeted_kwargs
