from pathlib import Path
from typing import Union, Optional
import pandas as pd

import torch
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchok.data.datasets.base import ImageDataset
from torchvision.datasets.utils import download_and_extract_archive

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
    tgz_md5 = '26513716999698fd361a21c93f77ed32'

    train_txt = 'Ebay_train.txt'
    test_txt = 'Ebay_test.txt'

    def __init__(self,
                 train: bool,
                 download: bool,
                 data_folder: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 image_dtype: str = 'float32',
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init SOP.

        Have 120,053 images with 22,634 classes in the dataset in total. 
        Train have 59551 images with 11318 classes.
        Test have 60502 images with 11316 classes.

        Args:
            train: If True, train dataset will be used, else - test dataset.
            download: If True, data will be download and save to data_folder.
            data_folder: Directory with all the images.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            image_dtype: Data type of of the torch tensors related to the image.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
        """
        super().__init__(transform, augment, image_dtype, grayscale, test_mode)
        self.__data_folder = Path(data_folder)
        self.__path = self.__data_folder / self.base_folder
        self.__train = train

        if download:
            self.__download()

        if not self.__path.is_dir():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        if self.__train:
            self.__csv = pd.read_csv(self.__path / self.train_txt, sep=' ')
        else:
            self.__csv = pd.read_csv(self.__path / self.test_txt, sep=' ')

        self.__target_column = 'class_id'
        self.__path_column = 'path'

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            sample: dict, where
            sample['image'] - Tensor, representing image after augmentations and transformations, dtype=image_dtype.
            sample['target'] - Target class or labels, dtype=target_dtype.
            sample['index'] - Index.
        """
        record = self.__csv.iloc[idx]
        image = self._read_image(self.__path / record[self.__path_column])
        sample = {"image": image}
        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)
        sample['image'] = sample['image'].type(torch.__dict__[self._image_dtype])
        sample['index'] = idx

        if self._test_mode:
            return sample

        if self.__train:
            # The labels starts with 1 for train
            sample['target'] = record[self.__target_column] - 1
        else:
            # The labels starts with 11319 for train
            sample['target'] = record[self.__target_column] - 11319

        return sample

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.__csv)

    def __download(self) -> None:
        """Download archive by url to specific folder."""
        if self.__path.is_dir():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.url, self.__data_folder, filename=self.filename, md5=self.tgz_md5)
