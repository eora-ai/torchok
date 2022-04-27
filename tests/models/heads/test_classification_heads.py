import unittest

import torch
#from src.models.heads.classification.arcfacehead import ArcFaceHead
from src.constructor import CLASSIFICATION_HEADS, HEADS



class TestArcFaceHead(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)

    def test_len(self):
        self.__in_features = 128
        self.__num_classes = 10
        self.__arcface = HEADS.get('ArcFaceHead')(self.__in_features, self.__num_classes)
        

