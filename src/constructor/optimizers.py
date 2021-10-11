from collections.abc import Iterable

import torch
import torch.nn as nn

from src.constructor.config_structure import StructureParams
from src.optim.optimizers.lookahead import Lookahead
from src.registry import OPTIMIZERS, SCHEDULERS


def create_optimizer(model, optim_param: StructureParams):
    optimizer_name = optim_param.name.split('_')

    optimizer_class = OPTIMIZERS.get(optimizer_name[-1])
    parameters = get_parameters(model)
    optimizer = optimizer_class(parameters, **optim_param.params)

    if len(optimizer_name) > 1 and optimizer_name[0].lower() == 'lookahead':
        optimizer = Lookahead(optimizer)

    return optimizer


def create_scheduler(optimizer, param: StructureParams):
    scheduler_class = SCHEDULERS.get(param.name)
    scheduler = scheduler_class(optimizer, **param.params)

    # In case Pytorch Lightning parameters are specified return dictionary containing scheduler and the parameters
    if param.aux_params:
        scheduler = {
            'scheduler': scheduler
        }
        scheduler.update(param.aux_params)

    return scheduler


def get_parameters(model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    if isinstance(model, nn.Module):
        if len(list(model.parameters())) > 0:
            return set_weight_decay(model)
        else:
            return []
    elif isinstance(model, Iterable):
        param_groups = []
        parameters = []
        for part in model:
            if isinstance(part, nn.Module) and len(list(part.parameters())) == 0:
                continue

            if isinstance(part, nn.Module):
                param_groups.extend(set_weight_decay(part))
            elif isinstance(part, torch.Tensor) and part.requires_grad:
                parameters.append(part)
        if len(parameters) > 0:
            param_groups.append({'params': parameters})

        return param_groups
    else:
        return []


def set_weight_decay(model):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
        else:
            has_decay.append(param)
    output = []
    if has_decay:
        output.append({'params': has_decay})
    if no_decay:
        output.append({'params': no_decay, 'weight_decay': 0.})
    return output


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
