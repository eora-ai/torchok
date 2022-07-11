from typing import Dict, Union, Any, List

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

        self.__input_names = [input_name.name for input_name in self._sess.get_inputs()]
        self.__input_shapes = [item['shape'] for item in self.__infer_params.inputs]

        dynamic_dim = self.__input_shapes[0][0]

        self.__label_name = self._sess.get_outputs()[0].name
        self.__output_shape = self._sess.get_outputs()[0].shape
        self.__dynamic_output_shape = (dynamic_dim, *self.__output_shape[1:])

    def forward(self, x: Tensor) -> Tensor:
        pass

    def forward_with_gt(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        pass

    def foward_infer(self, x: Tensor) -> Tensor:
        """Foward inference."""
        output = self._forward_onnx(x)
        return output

    def _forward_onnx(self, inputs: Union[List[Tensor], Tensor]) -> Tensor:
        """Forward onnx model."""

        for input_name, input_shape, input_tensor in zip(self.__input_names, self.__input_shapes, inputs):
            self.__binding.bind_input(
                name=input_name,
                device_type=self.device,
                device_id=self.device_id,
                element_type=np.float32,
                shape=input_shape,
                buffer_ptr=input_tensor.data_ptr())

        output_tensor = torch.empty(self.__dynamic_output_shape, dtype=torch.float32, device=self.device).contiguous()

        self.__binding.bind_output(
            name=self.__label_name,
            device_type=self.device,
            device_id=self.device_id,
            element_type=np.float32,
            shape=self.__dynamic_output_shape,
            buffer_ptr=output_tensor.data_ptr())

        self._sess.run_with_iobinding(self.__binding)
        return output_tensor

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
        output = self.foward_infer(batch['image'])
        return output
