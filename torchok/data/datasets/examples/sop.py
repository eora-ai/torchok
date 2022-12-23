from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchvision.datasets.utils import download_and_extract_archive

from torchok.data.datasets.base import ImageDataset
from torchok.constructor import DATASETS


@DATASETS.register_class
class SOP(ImageDataset):
    """A class represent Stanford Online Products - SOP dataset.

    Additionally, we collected Stanford Online Products dataset: 120k images of 23k classes of online products
    for metric learning. The homepage of SOP is https://cvgl.stanford.edu/projects/lifted_struct/.
    """
    base_folder = 'Stanford_Online_Products'
    filename = 'Stanford_Online_Products.tar.gz'

    url = 'https://torchok-hub.s3.eu-west-1.amazonaws.com/Stanford_Online_Products.tar.gz'
    tgz_md5 = 'b96128cf2b75493708511ff5c400eefe'

    train_txt = 'Ebay_train.txt'
    test_txt = 'Ebay_test.txt'

    def __init__(self,
                 train: bool,
                 download: bool,
                 data_folder: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 channel_order: str = 'rgb',
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init SOP.

        Have 120,053 images with 22,634 classes in the dataset in total.
        Train have 59551 images with 11318 classes.
        Test have 60502 images with 11316 classes.

        Args:
            train: If True, train dataset will be used, else - test dataset.
            download: If True, data will be downloaded and save to data_folder.
            data_folder: Directory with all the images.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of the torch tensors related to the image.
            channel_order: Order of channel, candidates are `bgr` and `rgb`.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
        """
        super().__init__(
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            channel_order=channel_order,
            grayscale=grayscale,
            test_mode=test_mode
        )
        self.data_folder = Path(data_folder)
        self.path = self.data_folder / self.base_folder
        self.train = train

        if download:
            self._download()

        if not self.path.is_dir():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        txt = self.train_txt if self.train else self.test_txt
        self.csv = pd.read_csv(self.path / txt, sep=' ')

        self.target_column = 'class_id'
        self.path_column = 'path'

    def get_raw(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations.
            sample['target'] - Target class or labels.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        record = self.csv.iloc[idx]
        image = self._read_image(self.path / record[self.path_column])
        sample = {"image": image, 'index': idx}

        if not self.test_mode:
            if self.train:
                # The labels start with 1 for train
                sample['target'] = record[self.target_column] - 1
            else:
                # The labels start with 11319 for train
                sample['target'] = record[self.target_column] - 11319

        sample = self._apply_transform(self.augment, sample)

        return sample

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=input_dtype.
            sample['target'] - Target class or labels.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        sample = self.get_raw(idx)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        return sample

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.csv)

    def _download(self) -> None:
        """Download archive by url to specific folder."""
        if self.path.is_dir():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.url, self.data_folder.as_posix(),
                                         filename=self.filename, md5=self.tgz_md5)
