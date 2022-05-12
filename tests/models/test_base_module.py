import unittest

import torch
import torch.nn as nn

from typing import List

from src.models.base_model import BaseModel, FeatureInfo


class TestBaseModelWithoutHooks(unittest.TestCase):
    def setUp(self):
        self._input_channel = 3
        self._output_channel = 5
        self._input = torch.ones(1, self._input_channel, 224, 224)

        class ModelWithoutHooks(BaseModel):
            def __init__(self, input_channel: int, output_channel: int):
                super().__init__()
                self.conv = nn.Conv2d(input_channel, output_channel, 3, padding='same')
                self.output_channel = output_channel

            def get_forward_output_channels(self):
                # Impltment abstract method
                return self.output_channel

            def forward(self, x):
                return self.conv(x)

        self._model = ModelWithoutHooks(self._input_channel, self._output_channel)

    def test_function_get_out_channel_when_hooks_was_not_define(self):
        print('test_function_get_out_channel_when_hooks_was_not_define')
        forward_channels, hooks_channels = self._model.get_output_channels()
        self.assertEqual(forward_channels, self._output_channel, 'Forward channels is not equal.')
        self.assertEqual(hooks_channels, None, 'Hooks channels is not None.')

    def test_forward_stage_features_when_hooks_was_not_define(self):
        print('test_function_get_out_channel_when_hooks_was_not_define')
        last_features, hooks_features = self._model.forward_stage_features(self._input)
        feature_shape = (1, self._output_channel, 224, 224)
        self.assertEqual(last_features.shape, feature_shape, 'Forward channels shape is not equal.')
        self.assertEqual(hooks_features, None, 'Hooks channels is not None.')


class TestBaseModelWithHooks(unittest.TestCase):
    def setUp(self):
        self._input_channel = 3
        self._output_channels = [5, 10, 15]
        self._input = torch.ones(1, self._input_channel, 224, 224)

        class ModelWithHooks(BaseModel):
            def __init__(self, input_channel: int, output_channels: List[int]):
                super().__init__()
                conv_list = []
                for i in range(len(output_channels)):
                    if i == 0:
                        conv_list.append(nn.Conv2d(input_channel, output_channels[i], 3, padding='same'))
                    else:
                        conv_list.append(nn.Conv2d(output_channels[i-1], output_channels[i], 3, padding='same'))
                self.conv_list = nn.ModuleList(conv_list)
                self.output_channels = output_channels
                self.create_hooks()

            def get_features_info(self):
                features_info = []
                for i in range(len(self.output_channels)):
                    module_name = 'conv_list.' + str(i)
                    num_channels = self.output_channels[i]
                    feature_info = FeatureInfo(module_name=module_name, num_channels=num_channels, stride=1)
                    features_info.append(feature_info)
                return features_info

            def get_forward_output_channels(self):
                # Impltment abstract method
                return self.output_channels[-1]

            def forward(self, x):
                for i in range(len(self.conv_list)):
                    x = self.conv_list[i](x)
                return x

        self._model = ModelWithHooks(self._input_channel, self._output_channels)

    def test_hooks_output_channels_when_hooks_was_define(self):
        print('test_function_get_out_channel_when_hooks_was_not_define')
        forward_channels, hooks_channels = self._model.get_output_channels()
        print(f'hooks channels = {hooks_channels}')
        self.assertEqual(forward_channels, self._output_channels[-1], 'Forward channels is not equal.')
        self.assertListEqual(hooks_channels, self._output_channels, 'Hooks channels is not equal.')
        
    def test_forward_stage_features_when_hooks_was_define(self):
        print('test_forward_stage_features_when_hooks_was_define')
        last_features, hooks_features = self._model.forward_stage_features(self._input)
        hooks_features_shapes_answer = [(1, 5, 224, 224), (1, 10, 224, 224), (1, 15, 224, 224)]
        hooks_features_shapes = []
        for hook in hooks_features:
            hooks_features_shapes.append(hook.shape)
        self.assertEqual(last_features.shape, hooks_features_shapes_answer[-1], 'Forward channels shape is not equal.')
        self.assertListEqual(hooks_features_shapes, hooks_features_shapes_answer, 'Hooks channels shapes is not equal.')
    

class TestDocumentationExample(unittest.TestCase):
    class ModelWithHooks(BaseModel):
        def __init__(self, input_channel: int = 3):
            super().__init__()
            self.out_features = [5, 10, 15]
            self.conv1 = nn.Conv2d(input_channel, 5, 3)
            self.conv2 = nn.Conv2d(5, 10, 3)
            self.conv3 = nn.Conv2d(10, 15, 3)
            # Call _create_hooks after all modules is initialize, with parameters as in _get_features_info.
            self.create_hooks(output_channels=[10, 15], module_names=['conv2', 'conv3'])

        def get_features_info(self, output_channels, module_names):
            # Overwrite get_features_info.
            features_info = []
            for num_channels, module_name in zip(output_channels, module_names):
                feature_info = FeatureInfo(module_name=module_name, num_channels=num_channels, stride=1)
                features_info.append(feature_info)
            return features_info

        def get_forward_output_channels(self):
            return self.out_features[-1]

    _output_channels = [10, 15]
    _model = ModelWithHooks()

    def test_hooks_output_channels_when_args_was_defined(self):
        print('test_function_get_out_channel_when_hooks_was_not_define')
        forward_channels, hooks_channels = self._model.get_output_channels()
        print(self._model.get_output_channels())
        self.assertEqual(forward_channels, self._output_channels[-1], 'Forward channels is not equal.')
        self.assertListEqual(hooks_channels, self._output_channels, 'Hooks channels is not equal.')
    