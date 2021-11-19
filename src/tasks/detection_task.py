import torch
from pydantic import BaseModel

from src.constructor import create_backbone, create_scheduler, create_optimizer
from src.constructor.config_structure import TrainConfigParams
from src.registry import TASKS, DETECTION_NECKS, DETECTION_HEADS
from .base_task import BaseTask

class DetectionTaskParams(BaseModel):
    checkpoint: str = None
    input_size: list  # [num_channels, height, width]
    freeze_backbone: bool = False
    backbone_name: str
    backbone_params: dict = {}
    neck_name: str
    neck_params: dict = {}
    head_name: str
    head_params: dict = {}


@TASKS.register_class
class DetectionTask(BaseTask):
    config_parser = DetectionTaskParams

    def __init__(self, hparams: TrainConfigParams):
        super().__init__(hparams)
        self.backbone = create_backbone(model_name=self.params.backbone_name,
                                        **self.params.backbone_params)

        neck_class = DETECTION_NECKS.get(self.params.neck_name)
        self.neck = neck_class(
                            in_channels=[128, 256, 512],
                            out_channels=128,
                            num_csp_blocks=1
        )
        
        head_class = DETECTION_HEADS.get(self.params.head_name)
        self.head = head_class(
                            num_classes=2, 
                            in_channels=128, 
                            feat_channels=128
        )
   

    def forward(self, x):
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        output = self.head(neck_features)
        return output
