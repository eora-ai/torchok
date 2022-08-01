import logging
from typing import Dict, Union, Any

import onnx
import torch
from torch import Tensor
import onnxruntime as onnxrt
from omegaconf import DictConfig

from torchok.constructor import TASKS
from torchok.tasks.base import BaseTask
from torchok.constructor.config_structure import Phase


@TASKS.register_class
class ONNXTask(BaseTask):
    """A class for onnx task."""

    str_type2numpy_type = {'tensor(float)': 'float32', 'tensor(float16)': 'float16',
                           'tensor(double)': 'float64', 'tensor(int8)': 'int8',
                           'tensor(int16)': 'int16', 'tensor(int32)': 'int32',
                           'tensor(int64)': 'int64', 'tensor(uint8)': 'uint8'}

    def __init__(self, hparams: DictConfig):
        """Init ONNXTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)
        self.__infer_params = self._hparams.task.params
        model_path = self.__infer_params.path_to_onnx

        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)

        self._sess = onnxrt.InferenceSession(model_path, providers=self.__infer_params.providers)
        self.__binding = self._sess.io_binding()

        self.__inputs = [{'name': item.name,
                          'dtype': self.str_type2numpy_type[item.type]} for item in self._sess.get_inputs()]

        input_names = [input['name'] for input in self.__inputs]
        logging.info(f'ONNX model input names: {input_names}')

        self.__keys_mapping_onnx2dataset = self._hparams.task.params.keys_mapping_onnx2dataset

        self.__outputs = [{'name': item.name,
                           'shape': item.shape,
                           'dtype': self.str_type2numpy_type[item.type]} for item in self._sess.get_outputs()]

        output_names = [output['name'] for output in self.__outputs]
        logging.info(f'ONNX model output names: {output_names}')

    def forward(self, x: Tensor) -> Tensor:
        pass

    def forward_with_gt(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        pass

    def foward_infer(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward onnx model."""

        for input in self.__inputs:
            # TODO: Hardcode device_id check that it doesn't matter
            input_tensor = inputs[self.__keys_mapping_onnx2dataset[input['name']]]
            self.batch_dim = input_tensor.shape[0]

            self.__binding.bind_input(
                name=input['name'],
                device_type=self.device,
                device_id=0,
                element_type=input['dtype'],
                shape=input_tensor.shape,
                buffer_ptr=input_tensor.data_ptr())

        output = dict()

        for output_params in self.__outputs:
            output_tensor = torch.empty((self.batch_dim, *output_params['shape'][1:]),
                                        dtype=torch.__dict__[output_params['dtype']],
                                        device=self.device).contiguous()
            self.__binding.bind_output(
                name=output_params['name'],
                device_type=self.device,
                device_id=0,
                element_type=output_params['dtype'],
                shape=(self.batch_dim, *output_params['shape'][1:]),
                buffer_ptr=output_tensor.data_ptr())

            output[output_params['name']] = output_tensor

        self._sess.run_with_iobinding(self.__binding)
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
        self._metrics_manager.forward(Phase.TEST, **output)

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """Complete predict loop."""
        output = self.foward_infer(batch)
        return output
