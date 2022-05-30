import unittest

import torch
from src.constructor import HEADS


class TestArcFaceHead(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)

    def test_shape(self):
        self.__in_features = 128
        self.__num_classes = 10
        self.__arcface = HEADS.get('ArcFaceHead')(self.__in_features, self.__num_classes)
        self.__input = torch.rand((2,self.__in_features))
        self.__target = torch.tensor([[4],[8]])
        output = self.__arcface(self.__input, self.__target)
        self.assertEqual(output.shape, (2, 10))
    
    def test_weight_shape(self):
        self.__in_features = 128
        self.__num_classes = 10
        self.__arcface = HEADS.get('ArcFaceHead')(self.__in_features, self.__num_classes)
        self.assertEqual(self.__arcface.weights.shape, (self.__num_classes, self.__in_features))
