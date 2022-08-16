from pathlib import Path
from typing import Union, Optional

from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchok.data.datasets.classification.classification import ImageClassificationDataset
from torchvision.datasets.utils import download_and_extract_archive

from torchok.constructor import DATASETS


@DATASETS.register_class
class MULTILABEL_TOILETS(ImageClassificationDataset):
    """A class represent multilabel pairwise dateset - Toilets dataset."""
    base_folder = 'toilets'
    filename = 'toilets.tar.gz'

    url = 'https://torchok-hub.s3.eu-west-1.amazonaws.com/toilets.tar.gz'
    tgz_md5 = 'cfb02ce117a775f31e784d9ce76e890c'

    train_csv = 'multilabel_train.csv'

    def __init__(self,
                 train: bool,
                 download: bool,
                 data_folder: str,
                 num_classes: int,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 image_dtype: str = 'float32',
                 target_dtype: str = 'int64',
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init MULTILABEL_TOILETS.

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
        self.__multilabel = True
        self.__data_folder = Path(data_folder)
        self.__path = self.__data_folder / self.base_folder

        if download:
            self.__download()

        if not self.__path.is_dir():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        csv_path = self.train_csv

        super().__init__(
            data_folder=self.__path,
            csv_path=csv_path,
            num_classes=num_classes,
            transform=transform,
            augment=augment,
            image_dtype=image_dtype,
            target_dtype=target_dtype,
            grayscale=grayscale,
            test_mode=test_mode,
            multilabel=self.__multilabel
        )

    def __download(self) -> None:
        """Download archive by url to specific folder."""
        if self.__path.is_dir():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.url, self.__data_folder, filename=self.filename, md5=self.tgz_md5)
