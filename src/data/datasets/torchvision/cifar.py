import random
from pathlib import Path
import pickle
from typing import Any, Union, Optional

import numpy as np
import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from src.data.datasets.abc_dataset import ABCDataset
from src.registry import DATASETS
from ..multi import random_range


@DATASETS.register_class
class CIFAR10(ABCDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string, Path): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable): Transformation that takes in a numpy image and returns a transformed version
                This should have the interface of transforms in `albumentations` library.
        augment (callable, optional): Augmentation that takes in a numpy image and returns a transformed version
                This should have the interface of transforms in `albumentations` library.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        input_dtype (str, optional): data type of of the torch tensors related to the image
        test_mode (bool, optional): if True, only image without labels will be returned

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
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

    def __init__(self, root: str, train: str, transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None, download=False,
                 input_dtype: str = 'float32', test_mode: Optional[bool] = False) -> None:

        super().__init__(transform, augment)
        self.root = Path(root)
        self.train = train  # training set or test set
        self.input_dtype = input_dtype
        self.test_mode = test_mode

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = self.root / self.base_folder / file_name
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.targets = np.array(self.targets, dtype=np.int64)
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self.update_transform_targets({'input': 'image'})

    def _load_meta(self) -> None:
        path = self.root / self.base_folder / self.meta['filename']
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, idx: int) -> dict:
        sample = self.get_raw(idx)
        sample = self.apply_transform(self.transform, sample)
        sample['input'] = sample['input'].type(torch.__dict__[self.input_dtype])
        return sample

    def __len__(self) -> int:
        return len(self.data)

    def get_raw(self, idx: int) -> dict:
        image = self.data[idx]
        sample = {"input": image}
        sample = self.apply_transform(self.augment, sample)

        if not self.test_mode:
            sample["target"] = self.targets[idx]

        return sample

    def _check_integrity(self) -> bool:
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = self.root / self.base_folder / filename
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


@DATASETS.register_class
class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


@DATASETS.register_class
class ContrastiveCIFAR10(CIFAR10):
    def __init__(self,
                 root,
                 train: str,
                 transform: Union[BasicTransform, BaseCompose],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 download: bool = False,
                 input_dtype: str = 'float32',
                 return_targets: bool = False,
                 sup_part: float = 0.01,
                 sup_part_determined: bool = False,
                 num_sup_epochs: int = 0):
        super().__init__(root, train, transform, augment, download, input_dtype)

        self.sup_part = sup_part if isinstance(sup_part, float) else (sup_part / len(self.data))
        if sup_part_determined:
            self.is_sup = np.arange(self.targets.shape[0]) < (self.sup_part * self.targets.shape[0])
        else:
            self.is_sup = np.random.random(self.targets.shape) < self.sup_part
        self.return_targets = return_targets
        self.num_sup_epochs = num_sup_epochs

    def __getitem__(self, idx: int) -> dict:
        sample = self.get_raw(idx)
        sample_0 = self.apply_transform(self.transform, {'input': sample['input_0']})
        sample_1 = self.apply_transform(self.transform, {'input': sample['input_1']})
        sample['input_0'] = sample_0['input'].type(torch.__dict__[self.input_dtype])
        sample['input_1'] = sample_1['input'].type(torch.__dict__[self.input_dtype])

        return sample

    def __len__(self) -> int:
        return len(self.data)

    def get_raw(self, idx: int) -> dict:
        sample = {"input": self.data[idx]}
        sample_0 = self.apply_transform(self.augment, sample)['input']
        sample_1 = self.apply_transform(self.augment, sample)['input']
        sample = {'input_0': sample_0, 'input_1': sample_1}
        if self.return_targets:
            sample['target'] = self.targets[idx]
            sample['is_sup'] = self.is_sup[idx]

        return sample

    def batch_sampler(self, batch_size, shuffle=False, drop_last=False):
        return CustomBatchSampler(batch_size, drop_last, shuffle, self)


class CustomBatchSampler:
    r"""Wraps Contrastive dataset to yield a mini-batch of indices. From contrastive dataset it gets
    ``num_sup_epochs`` during which it returns only samples that has labels. And then returns all samples
    in the dataset

    Args:
        dataset (ContrastiveCIFAR10): an instance of ContrastiveCIFAR10 dataset
        batch_size (int): Size of mini-batch.
        shuffle (bool): if ``True``, indices of aligned dataset will be shuffled.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    def __init__(self, batch_size, drop_last, shuffle, dataset):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset
        self.shuffle = shuffle

        self.num_sup_epochs = dataset.num_sup_epochs
        self.cur_epoch = 0

    def __iter__(self):
        if self.cur_epoch >= self.num_sup_epochs:
            dataset_len = len(self.dataset)
            idxs = random_range(dataset_len) if self.shuffle else iter(range(dataset_len))
        else:
            idxs = np.nonzero(self.dataset.is_sup)[0]
            if self.shuffle:
                random.shuffle(idxs)

        batch = []
        for idx in idxs:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
        self.cur_epoch += 1

    def __len__(self):
        dataset_len = len(self.dataset) if self.cur_epoch >= self.num_sup_epochs else self.dataset.is_sup.sum()
        if self.drop_last:
            return dataset_len // self.batch_size  # type: ignore
        else:
            return (dataset_len + self.batch_size - 1) // self.batch_size  # type: ignore
