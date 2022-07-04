import unittest

import torch
import copy
from typing import Dict, List, Optional
from collections import OrderedDict

from .checkpoints.model import Model
from src.constructor.load import load_checkpoint, generate_required_state_dict


def load_state_dict_with_prefix(state_dict_path: str, prefix: str) -> Dict[str, torch.Tensor]:
    """Load state dict with prefixed keys.
    Args:
        state_dict_path: Loaded state dict path.
        prefix: Prefix for keys.
    Returns:
        prefix_state_dict: Loaded state dict from state_dict_path with prefix in dictionary keys.
    """
    state_dict = torch.load(state_dict_path)
    prefix_state_dict = OrderedDict()
    for key, value in state_dict.items():
        prefix_state_dict[prefix + '.' + key] = value
    return prefix_state_dict


def compare_state_dicts(current_state_dict: Dict[str, torch.Tensor], load_state_dict: Dict[str, torch.Tensor], 
                        check_keys: Optional[List[str]] = None):
    """Compare state dicts by check_keys. Assert if check keys is not equal.

    Args:
        current_state_dict: First compared state dict.
        load_state_dict: Second compared state dict.
        check_keys: The keys for compare.
    """
    check_keys = list(current_state_dict.keys()) if check_keys is None else check_keys
    for key in check_keys:
        assert torch.equal(current_state_dict[key], load_state_dict[key])


class TestCheckpoint(unittest.TestCase):
    base_path = 'tests/constructor/checkpoints/base.pth'
    layer_path = 'tests/constructor/checkpoints/layer.pth'

    model_keys = [
        'layer1.module.conv1.weight', 
        'layer1.linear.weight',
        'linear.weight',
    ]

    initial_state_dict = {
        'layer1.module.conv1.weight': 0, 
        'layer1.linear.weight': 0,
        'linear.weight': 0,
    }

    base_state_dict = {
        'layer1.module.conv1.weight': 1, 
        'layer1.linear.weight': 2,
        'linear.weight': 3,
    }

    def test_generate_required_state_dict_when_only_base_checkpoint_was_defined(self):

        overridden_state_dict = dict()
        exclude_keys = list()
        answer_state_dict = self.base_state_dict
        
        generated_state_dict = generate_required_state_dict(self.base_state_dict, overridden_state_dict, 
                                                            exclude_keys, self.model_keys, self.initial_state_dict)

        self.assertDictEqual(answer_state_dict, generated_state_dict)

    def test_generate_required_state_dict_when_base_and_overridden_checkpoints_was_defined(self):
        overridden_state_dict = {
            'layer1': {
                'layer1.module.conv1.weight': 11,
                'layer1.linear.weight': 22
            }
        }

        exclude_keys = list()

        answer_state_dict = {
            'layer1.module.conv1.weight': 11, 
            'layer1.linear.weight': 22,
            'linear.weight': 3,
        }

        generated_state_dict = generate_required_state_dict(self.base_state_dict, overridden_state_dict, 
                                                            exclude_keys, self.model_keys, self.initial_state_dict)

        self.assertDictEqual(answer_state_dict, generated_state_dict)

    def test_generate_required_state_dict_when_full_parameters_was_defined(self):
        overridden_state_dict = {
            'layer1': {
                'layer1.module.conv1.weight': 11,
                'layer1.linear.weight': 22
            }
        }

        exclude_keys = ['layer1.module']

        answer_state_dict = {
            'layer1.linear.weight': 22,
            'linear.weight': 3,
            'layer1.module.conv1.weight': 0 
        }

        generated_state_dict = generate_required_state_dict(self.base_state_dict, overridden_state_dict, 
                                                            exclude_keys, self.model_keys, self.initial_state_dict)

        self.assertDictEqual(answer_state_dict, generated_state_dict)

    def test_generate_required_state_dict_when_overridden_state_dict_had_intersection_keys(self):
        overridden_state_dict = {
            'layer1': {
                'layer1.module.conv1.weight': 11,
                'layer1.linear.weight': 22
            },
            'layer1.linear': {
                'layer1.linear.weight': 222
            }
        }

        exclude_keys = ['layer1.module']

        answer_state_dict = {
            'layer1.linear.weight': 222,
            'linear.weight': 3,
            'layer1.module.conv1.weight': 0
        }

        generated_state_dict = generate_required_state_dict(self.base_state_dict, overridden_state_dict, 
                                                            exclude_keys, self.model_keys, self.initial_state_dict)

        self.assertDictEqual(answer_state_dict, generated_state_dict)

    def test_checkpoint_load_when_full_parameters_was_defined(self):
        model = Model()
        current_state_dict = copy.deepcopy(model.state_dict())
        override_name2path = {
            'layer': self.layer_path
        }
        exclude_keys = ['layer.block2']

        load_checkpoint(model, self.base_path, override_name2path, exclude_keys)

        # Partition comparing
        loaded_state_dict = model.state_dict()

        # names which not loaded
        full_exclude_keys = [
            'layer.block2.conv.weight', 'layer.block2.conv.bias', 
            'layer.block2.linear.weight', 'layer.block2.linear.bias', 
        ]
        # Names which load from base checkpoints
        full_base_keys = ['linear.weight', 'linear.bias']
        # Names which load from override checkpoints
        full_overridden_keys = ['layer.linear.weight', 'layer.linear.bias']

        # Compare weights for base checkpoint i.e which not in override keys and not in exclude keys
        base_state_dict = torch.load(self.base_path)
        compare_state_dicts(base_state_dict, loaded_state_dict, check_keys=full_base_keys)

        # Compare weights in override_dict
        overridden_state_dict = load_state_dict_with_prefix(self.layer_path, 'layer')
        compare_state_dicts(overridden_state_dict, loaded_state_dict, check_keys=full_overridden_keys)

        # Compare weights exclude keys
        compare_state_dicts(current_state_dict, loaded_state_dict, check_keys=full_exclude_keys)

    def test_checkpoint_load_when_base_checkpoint_was_not_full(self):
        model = Model()

        base_path = self.layer_path
        full_overridden_keys = dict() 
        exclude_keys = list()

        with self.assertRaises(Exception):
            load_checkpoint(model, base_path, full_overridden_keys, exclude_keys)

    def test_checkpoint_load_when_overridden_key_was_not_in_model_state_dict(self):
        model = Model()
        full_overridden_keys = {'loyer': self.layer_path}
        exclude_keys = ['layer.block2']

        with self.assertRaises(Exception):
            load_checkpoint(model, self.base_path, full_overridden_keys, exclude_keys)

    def test_checkpoint_load_when_exclude_key_was_not_right(self):
        model = Model()
        full_overridden_keys = {
            'layer': self.layer_path
        }
        exclude_keys = ['loyer.block2']

        with self.assertRaises(Exception):
            load_checkpoint(model, self.base_path, full_overridden_keys, exclude_keys)
    