import unittest
from pathlib import Path

import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import Resize, HorizontalFlip, VerticalFlip, Normalize

from torchok.constructor import DATASETS

class TestClassificationMulticlass(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.__root_dir = Path(__file__).parent
        self.__data_folder = self.__root_dir / 'cifar10'
        self.__train = True
        self.__download = True
        self.__transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ToTensorV2()], p=1.0)
        self.__augment = None

    def test_len(self):
        self.__ds = DATASETS.get('CIFAR10')(self.__train,
                            self.__download,
                            self.__data_folder,
                            self.__transform,
                            self.__augment)
        print(self.__ds[1]['image'].shape)
        self.assertEqual(len(self.__ds), 50000)