from pathlib import Path
from typing import Union, Optional, Dict
import pandas as pd

import torch
from torch import Tensor
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchok.data.datasets.base import ImageDataset
from torchvision.datasets.utils import download_and_extract_archive

from torchok.constructor import DATASETS


@DATASETS.register_class
class PAIRWISE_TOILETS(ImageDataset):
    """A class represent pairwise dateset - Toilets dataset."""
    base_folder = 'toilets'
    filename = 'toilets.tar.gz'

    url = 'https://torchok-hub.s3.eu-west-1.amazonaws.com/toilets.tar.gz'
    tgz_md5 = 'cfb02ce117a775f31e784d9ce76e890c'

    train_csv = 'train.csv'

    def __init__(self,
                 train: bool,
                 download: bool,
                 data_folder: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 image_dtype: str = 'float32',
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init PAIRWISE_TOILETS.

        Args:
            train: If True, train dataset will be used, else - test dataset.
            download: If True, data will be downloaded and save to data_folder.
            data_folder: Directory with all the images.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            image_dtype: Data type of the torch tensors related to the image.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
        """
        super().__init__(transform, augment, image_dtype, grayscale, test_mode)
        self.__data_folder = Path(data_folder)
        self.__path = self.__data_folder / self.base_folder

        if download:
            self.__download()

        if not self.__path.is_dir():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        self.__csv = pd.read_csv(self.__path / self.train_csv)

        self.__anchor_paths_column = 'anchor'
        self.__positive_paths_column = 'positive'
        self.__negative_paths_column = 'negative'

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            output: dict, where
            output['anchor'] - Anchor.
            output['positive'] - Positive.
            output['negative'] - Negative.
            sample['index'] - Index.
        """
        record = self.__csv.iloc[idx]

        output = {'anchor': self.__image_preparation(record, self.__anchor_paths_column),
                  'positive': self.__image_preparation(record, self.__positive_paths_column),
                  'negative': self.__image_preparation(record, self.__negative_paths_column),
                  'index': idx}

        return output
    
    def __image_preparation(self, record: pd.Series, column_name: str) -> Dict[str, Tensor]:
        image = self._read_image(self.__path / record[column_name])
        sample = {"image": image}
        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)
        image = sample['image'].type(torch.__dict__[self.image_dtype])
        return image

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.__csv)

    def __download(self) -> None:
        """Download archive by url to specific folder."""
        if self.__path.is_dir():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.url, self.__data_folder, filename=self.filename, md5=self.tgz_md5)
