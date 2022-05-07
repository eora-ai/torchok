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
                self.conv = nn.Conv2d(input_channel, output_channel, 3)
                self.output_channel = output_channel

            def _get_forward_output_channels(self):
                # Impltment abstract method
                return self.output_channel

            def forward(self, x):
                return self.conv(x)

        self._model = ModelWithoutHooks(self._input_channel, self._output_channel)

    def test_function_get_out_channel_when_hooks_was_not_define(self):
        print('test_function_get_out_channel_when_hooks_was_not_define')
        forward_channels, hooks_channels = self._model.get_output_channels()
        self.assertEqual(forward_channels, self._output_channel, 'Forward channels is not equal.')
        self.assertEqual(hooks_channels, None, 'Hooks channels is not equal.')


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
                        conv_list.append(nn.Conv2d(input_channel, output_channels[i], 3))
                    else:
                        conv_list.append(nn.Conv2d(output_channels[i-1], output_channels[i], 3))
                self.conv_list = nn.ModuleList(conv_list)
                self.output_channels = output_channels

            def get_features_info(self):
                features_info = []
                output_channels = [5, 10, 15]
                for i in range(len(output_channels)):
                    module_name = 'conv_list.' + str(i)
                    num_channels = output_channels[i]
                    feature_info = FeatureInfo(module_name=module_name, num_channels=num_channels, stride=1)
                    features_info.append(feature_info)
                return features_info

            def _get_forward_output_channels(self):
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
        self.assertEqual(forward_channels, self._output_channels[-1], 'Forward channels is not equal.')
        self.assertListEqual(hooks_channels, self._output_channels, 'Hooks channels is not equal.')
