from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from torch import Tensor
from torch.nn import Module, ModuleList

from src.constructor import LOSSES

# Losses parameters
@dataclass
class LossParams:
    """
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
    """
    name: str
    tag: str
    mapping: Dict[str, str]
    weight: Optional[float]
    params: Dict = field(default_factory=dict)
    
@dataclass
class JointLossParams:
    """
    Args:
        loss_params: List parameters for loss.
        normalize_weights: Either to normalize the weights, so that they sum up to 1 before loss value calculation.
    """
    loss_params: List[LossParams]
    normalize_weights: bool = True


class JointLoss(Module):
    """Represents wrapper for loss modules.

    This joined loss can be forwarded as a weighted sum of losses and can provide direct access to each loss.
    """
    def __init__(self, joint_loss_params: JointLossParams):
        """Init JointLoss.

        Reduction isn't applied, so the included loss modules are responsible for reduction

        Args:
            joint_loss_params: Include loss_params for create losses and normalize_weights bool value which indicates 
                whether to normalize losses weights.

        Raises:
            - ValueError: When only a few weights are specified but not for all the loss modules
        """
        super().__init__()
        loss_params = joint_loss_params.loss_params
        normalize_weights = joint_loss_params.normalize_weights

        self.__losses = []
        self.__tags = []
        self.__mappings = []
        self.__weights = []

        for cur_loss_params in loss_params:
            loss_module = LOSSES.get(cur_loss_params.name)(**cur_loss_params.params)
            self.__losses.append(loss_module)
            self.__tags.append(cur_loss_params.tag)
            self.__mappings.append(cur_loss_params.mapping)
            self.__weights.append(cur_loss_params.weight)

        self.__losses = ModuleList(self.__losses)
        self.__tag2loss = {tag: loss for tag, loss in zip(tags, self.__losses) if tag is not None}


        num_specified_weights = len(list(filter(lambda w: w is not None, self.__weights)))
        if num_specified_weights > 0 and num_specified_weights != len(self.__losses):
            raise ValueError('Loss weights must be either specified for each loss function or '
                             'not specified for any loss function')

        if num_specified_weights == 0:
            self.__weights = [1.] * len(self.__losses)

        if normalize_weights:
            self.__weights = [w / sum(self.__weights) for w in self.__weights]

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
        for loss_module, mapping, tag, weight in zip(self.__losses, self.__mappings, self.__tags, self.__weights):
            targeted_kwargs = self.__map_arguments(mapping, **kwargs)
            loss = loss_module(**targeted_kwargs)
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
        if tag in self.__tag2loss:
            return self.__tag2loss[tag]
        else:
            raise KeyError(f'Cannot access loss {tag}. You should tag your losses for direct access with a tag key')

    @staticmethod
    def __map_arguments(mapping: Dict[str, str], **model_outputs) -> Dict[str, Any]:
        targeted_kwargs = {}
        for target_arg, source_arg in mapping.items():
            if source_arg in model_outputs:
                targeted_kwargs[target_arg] = model_outputs[source_arg]
            else:
                raise ValueError(f'Cannot find {source_arg} for your mapping {target_arg} : {source_arg}. '
                                 f'You should either add {source_arg} output to your model or remove the mapping '
                                 f'from configuration')
        return targeted_kwargs
