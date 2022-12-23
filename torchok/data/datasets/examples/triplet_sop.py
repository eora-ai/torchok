from pathlib import Path
from typing import Union, Optional, Dict

import torch
import pandas as pd
from torch import Tensor
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchvision.datasets.utils import download_and_extract_archive

from torchok.data.datasets.base import ImageDataset
from torchok.constructor import DATASETS


@DATASETS.register_class
class TRIPLET_SOP(ImageDataset):
    """A class represent Stanford Online Products - SOP dataset.

    Additionally, we collected Stanford Online Products dataset: 120k images of 23k classes of online products
    for metric learning. The homepage of SOP is https://cvgl.stanford.edu/projects/lifted_struct/.
    """
    base_folder = 'Stanford_Online_Products'
    filename = 'Stanford_Online_Products.tar.gz'

    url = 'https://torchok-hub.s3.eu-west-1.amazonaws.com/Stanford_Online_Products.tar.gz'
    tgz_md5 = 'b96128cf2b75493708511ff5c400eefe'

    train_csv = 'sop_triplet_train.csv'
    test_csv = 'sop_triplet_test.csv'

    def __init__(self,
                 train: bool,
                 download: bool,
                 data_folder: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 anchor_column: str = 'anchor',
                 positive_column: str = 'positive',
                 negative_column: str = 'negative',
                 input_dtype: str = 'float32',
                 channel_order: str = 'rgb',
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init TRIPLET SOP.

        Dataset have 11319 image pair(anchor, positive, negative).

        Args:
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
        self.anchor_column = anchor_column
        self.positive_column = positive_column
        self.negative_column = negative_column
        self.train = train

        if download:
            self._download()

        if not self.path.is_dir():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        if self.train:
            self.csv = pd.read_csv(self.path / self.train_csv)
        else:
            self.csv = pd.read_csv(self.path / self.test_csv)

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['anchor'] - Anchor.
            sample['positive'] - Positive.
            sample['negative'] - Negative.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        sample = {'anchor': self._image_preparation(idx, self.anchor_column),
                  'positive': self._image_preparation(idx, self.positive_column),
                  'negative': self._image_preparation(idx, self.negative_column),
                  'index': idx}

        return sample

    def _image_preparation(self, idx: int, column_name: str, apply_transform: bool = True) -> Dict[str, Tensor]:
        record = self.csv.iloc[idx]
        image = self._read_image(self.path / record[column_name])
        sample = {"image": image}
        sample = self._apply_transform(self.augment, sample)

        if apply_transform:
            sample = self._apply_transform(self.transform, sample)
            image = sample['image'].type(torch.__dict__[self.input_dtype])

        return image

    def get_raw(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations.
            sample['target'] - Target class or labels.
            sample['index'] - Index of the sample, the same as input `idx`.
        """
        sample = {'anchor': self._image_preparation(idx, self.anchor_column, apply_transform=False),
                  'positive': self._image_preparation(idx, self.positive_column, apply_transform=False),
                  'negative': self._image_preparation(idx, self.negative_column, apply_transform=False),
                  'index': idx}

        return sample

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.csv)

    def _download(self) -> None:
        """Download archive by url to specific folder."""
        if self.path.is_dir():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.url, self.url.as_posix(), remove_finished=True,
                                         filename=self.filename, md5=self.tgz_md5)
