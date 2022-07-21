from typing import Dict, Union, Any

import onnx
import torch
import numpy as np
from torch import Tensor
import onnxruntime as onnxrt
from omegaconf import DictConfig

from src.constructor import TASKS
from src.tasks.base import BaseTask
from src.constructor.config_structure import Phase


@TASKS.register_class
class ONNXTask(BaseTask):
    """A class for onnx task."""

    def __init__(self, hparams: DictConfig):
        """Init ONNXTask.

        Args:
            hparams: Hyperparameters that set in yaml file.
        """
        super().__init__(hparams)
        self.__infer_params = self._hparams.task.params
        model_path = self.__infer_params.path_to_onnx
        self.device_id = self.__infer_params.device_id

        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)

        self._sess = onnxrt.InferenceSession(model_path, providers=self.__infer_params.providers)
        self.__binding = self._sess.io_binding()
        
        self.__str_type2numpy_type = {'tensor(float)': 'float32'}

        self.__inputs = [{'name': item.name,
                          'shape': params[1].shape,
                          'dtype': self.__str_type2numpy_type[item.type]}
                          for item, params in zip(self._sess.get_inputs(), self.__infer_params.inputs.items())]
        # # batch shape
        dynamic_dim = self.__inputs[0]['shape'][0]

        self.__outputs = [{'name': item.name,
                           'shape': (dynamic_dim, *item.shape[1:]),
                           'dtype': self.__str_type2numpy_type[item.type]} for item in self._sess.get_outputs()]

    def forward(self, x: Tensor) -> Tensor:
        pass

    def forward_with_gt(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        pass

    def foward_infer(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward onnx model."""
        # we leave only those inputs whose names match with the names from the config.
        input_tensors = [input_tensor for input_name, input_tensor in inputs.items() if input_name in self._input_names]
        
        for input, input_tensor in zip(self.__inputs, input_tensors):
            self.__binding.bind_input(
                name=input['name'],
                device_type=self.device,
                device_id=self.device_id,
                element_type=np.dtype(input['dtype']),
                shape=input['shape'],
                buffer_ptr=input_tensor.data_ptr())

        output = dict()

        for output_params in self.__outputs:
            output_tensor = torch.empty(output_params['shape'],
                                        dtype=torch.__dict__[output_params['dtype']],
                                        device=self.device).contiguous()
            self.__binding.bind_output(
                name=output_params['name'],
                device_type=self.device,
                device_id=self.device_id,
                element_type=output_params['dtype'],
                shape=output_params['shape'],
                buffer_ptr=output_tensor.data_ptr())

            output[output_params['name']] = output_tensor

        self._sess.run_with_iobinding(self.__binding)
        return output

    def forward_infer_with_gt(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Abstract forward method for test stage."""
        input_data = batch['image']
        target = batch['target']
        prediction = self.foward_infer(input_data)
        output = {'target': target, 'prediction': prediction}
        return output

    def test_step(self, batch: Dict[str, Union[Tensor, int]], batch_idx: int) -> None:
        """Complete test loop."""
        output = self.forward_infer_with_gt(batch)
        self._metrics_manager.forward(Phase.TEST, **output)

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        """Complete predict loop."""
        output = self.foward_infer(batch)
        return output
