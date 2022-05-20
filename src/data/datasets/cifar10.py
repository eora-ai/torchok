from pathlib import Path
from typing import Union, Optional, Any
import pickle
import numpy as np


import torch

from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import Compose
from albumentations import  Normalize
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from src.data.datasets.base import ImageDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from src.constructor import DATASETS


@DATASETS.register_class
class CIFAR10(ImageDataset):
    base_folder = 'cifar-10-batches-py'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = 'cifar-10-python.tar.gz'
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self,
                 train: str,
                 download: bool,
                 data_folder: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]] = None,
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 image_dtype: str = 'float32',
                 grayscale: bool = False,
                 test_mode: bool = False):
        if transform is None:
            transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ToTensorV2()], p=1.0)
        super().__init__(transform, augment, image_dtype, grayscale, test_mode)
        self.__data_folder = Path(data_folder)
        self.__train = train

        if download:
            self.__download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.__train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.__images = []
        self.__targets = []

        for file_name, _ in downloaded_list:
            file_path = self.__data_folder / self.base_folder / file_name
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.__images.append(entry['data'])
                if 'labels' in entry:
                    self.__targets.extend(entry['labels'])
                else:
                    self.__targets.extend(entry['fine_labels'])

        self.__targets = np.array(self.__targets, dtype=np.int64)
        self.__images = np.vstack(self.__images).reshape(-1, 3, 32, 32)
        self.__images = self.__images.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
    
    def _load_meta(self) -> None:
        path = self.__data_folder / self.base_folder / self.meta['filename']
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    
    def __getitem__(self, idx: int) -> dict:
        image = self.__images[idx]
        sample = {"image": image}
        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self._image_dtype])
        sample['index'] = idx

        if self._test_mode:
            return sample

        sample['target'] = self.__targets[idx]

        return sample

    def __len__(self) -> int:
        return len(self.__images)

    def _check_integrity(self) -> bool:
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = self.__data_folder / self.base_folder / filename
            if not check_integrity(fpath, md5):
                return False
        return True

    def __download(self) -> None:
        """Download archive by url to specific folder."""
        if self._check_integrity():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.url, self.__data_folder, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.__train is True else "Test")

