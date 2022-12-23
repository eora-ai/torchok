import logging
from typing import Any, Dict, Union

import onnx
import onnxruntime as onnxrt
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from torchok.constructor import TASKS
from torchok.constructor.config_structure import Phase
from torchok.tasks.base import BaseTask


@TASKS.register_class
class ONNXTask(BaseTask):
    """A class for onnx task."""

    str_type2numpy_type = {'tensor(float)': 'float32', 'tensor(float16)': 'float16',
                           'tensor(double)': 'float64', 'tensor(int8)': 'int8',
                           'tensor(int16)': 'int16', 'tensor(int32)': 'int32',
                           'tensor(int64)': 'int64', 'tensor(uint8)': 'uint8'}

    # ToDo: write documentation for the task parameters
    def __init__(self, hparams: DictConfig, path_to_onnx: str, providers, **kwargs):
        """Init ONNXTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
            path_to_onnx: path to ONNX model file.
            providers: Optional sequence of providers in order of decreasing
                precedence. Values can either be provider names or tuples of
                (provider name, options dict). If not provided, then all available
                providers are used with the default precedence.
        """
        super().__init__(hparams, **kwargs)
        onnx_model = onnx.load(path_to_onnx)
        onnx.checker.check_model(onnx_model)

        self.sess = onnxrt.InferenceSession(path_to_onnx, providers=providers)
        self.binding = self.sess.io_binding()

        self.inputs = [{'name': item.name,
                        'dtype': self.str_type2numpy_type[item.type]} for item in self.sess.get_inputs()]

        input_names = [inp['name'] for inp in self.inputs]
        logging.info(f'ONNX model input names: {input_names}')

        self.keys_mapping_onnx2dataset = self._hparams.task.params.keys_mapping_onnx2dataset

        self.outputs = [{'name': item.name,
                         'shape': item.shape,
                         'dtype': self.str_type2numpy_type[item.type]} for item in self.sess.get_outputs()]

        output_names = [output['name'] for output in self.outputs]
        logging.info(f'ONNX model output names: {output_names}')

    def forward(self, x: Tensor) -> Tensor:
        pass

    def forward_with_gt(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        pass

    def as_module(self) -> nn.Sequential:
        pass

    def foward_infer(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward onnx model."""

        for input in self.inputs:
            # TODO: Hardcode device_id check that it doesn't matter
            input_tensor = inputs[self.keys_mapping_onnx2dataset[input['name']]]
            self.batch_dim = input_tensor.shape[0]

            self.binding.bind_input(
                name=input['name'],
                device_type=self.device,
                device_id=0,
                element_type=input['dtype'],
                shape=input_tensor.shape,
                buffer_ptr=input_tensor.data_ptr())

        output = dict()

        for output_params in self.outputs:
            output_tensor = torch.empty((self.batch_dim, *output_params['shape'][1:]),
                                        dtype=torch.__dict__[output_params['dtype']],
                                        device=self.device).contiguous()
            self.binding.bind_output(
                name=output_params['name'],
                device_type=self.device,
                device_id=0,
                element_type=output_params['dtype'],
                shape=(self.batch_dim, *output_params['shape'][1:]),
                buffer_ptr=output_tensor.data_ptr())

            output[output_params['name']] = output_tensor

        self.sess.run_with_iobinding(self.binding)
        return output

    def forward_infer_with_gt(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Forward method for test stage."""
        target = batch['target']
        prediction = self.foward_infer(batch)
        output = {'target': target, **prediction}
        return output

    def test_step(self, batch: Dict[str, Union[Tensor, int]], batch_idx: int) -> None:
        """Complete test loop."""
        output = self.forward_infer_with_gt(batch)
        self.metrics_manager.update(Phase.TEST, **output)

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """Complete predict loop."""
        output = self.foward_infer(batch)
        return output
