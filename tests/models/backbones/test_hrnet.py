import unittest
import torch
from src.constructor import HEADS, NECKS, BACKBONES
import torch.nn.functional as F

class TestHRNetSegmentation(unittest.TestCase):

    def __init__(self, backbone_name, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._input = torch.ones(1, 3, 224, 224)
        self.backbone = BACKBONES.get(backbone_name)(pretrained=False, in_chans=3)
        neck_in_features = self.backbone.get_forward_output_channels()
        self.neck = NECKS.get('HRNetSegmentationNeck')(neck_in_features)
        head_in_features = self.neck.get_forward_output_channels()
        self.head = HEADS.get('HRNetSegmentationHead')(head_in_features, 10)
    def get_output(self, name):
        def hook(model, input, output):
            self._output[name] = output.detach()
        return hook

class TestHRNetSegmentation_W18(TestHRNetSegmentation):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('hrnet_w18', methodName)

    def test_outputs_equals(self):
        x = self.backbone(self._input)
        x = self.neck(x)
        x = self.head(x)
        print(x.shape)


class TestHRNetClassification(unittest.TestCase):

    def __init__(self, backbone_name, methodName: str = ...) -> None:
        super().__init__(methodName)
        self._input = torch.ones(1, 3, 224, 224)
        self.backbone = BACKBONES.get(backbone_name)(pretrained=False, in_chans=3)
        neck_in_features = self.backbone.get_forward_output_channels()
        self.neck = NECKS.get('HRNetClassificationNeck')(neck_in_features)
        head_in_features = self.neck.get_forward_output_channels()
        self.head = HEADS.get('ClassificationHead')(head_in_features, 10)
    def get_output(self, name):
        def hook(model, input, output):
            self._output[name] = output.detach()
        return hook

class TestHRNetClassification_W18(TestHRNetClassification):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__('hrnet_w18', methodName)

    def test_outputs_equals(self):
        x = self.backbone(self._input)
        x = self.neck(x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]).view(x.size(0), -1)
        x = self.head(x)
        print(x.shape)
