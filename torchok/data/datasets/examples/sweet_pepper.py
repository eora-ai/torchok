from pathlib import Path
from typing import Optional, Union

from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchvision.datasets.utils import download_and_extract_archive

from torchok.constructor import DATASETS
from torchok.data.datasets.segmentation.image_segmentation import ImageSegmentationDataset


@DATASETS.register_class
class SweetPepper(ImageSegmentationDataset):
    """A class represent segmentation dataset Sweet Pepper from Kaggle
    https://www.kaggle.com/datasets/lemontyc/sweet-pepper.

    The main task for this dataset is segment peppers (fruit) and peduncle on the images, obtained from different
    farm locations.
    Dataset has 3 labels: 0 - background, 1 - fruit and 2 - peduncle.
    Dataset contain 620 images in HD resolution, 500 - for train and 120 for validate.
    """
    base_folder = 'sweet_pepper'
    filename = 'sweet_pepper.tar.gz'

    url = 'https://torchok-hub.s3.eu-west-1.amazonaws.com/sweet_pepper.tar.gz'
    tgz_md5 = '65021e5fad5fe286b3c2bac7753d6e9d'

    train_csv = 'train.csv'
    valid_csv = 'valid.csv'

    def __init__(self,
                 train: bool,
                 download: bool,
                 data_folder: str,
                 transform: Optional[Union[BasicTransform, BaseCompose]],
                 augment: Optional[Union[BasicTransform, BaseCompose]] = None,
                 input_dtype: str = 'float32',
                 target_dtype: str = 'int64',
                 channel_order: str = 'rgb',
                 grayscale: bool = False,
                 test_mode: bool = False):
        """Init SweetPepper.

        Args:
            train: If True, train dataset will be used, else - test dataset.
            download: If True, data will be downloaded and save to data_folder.
            data_folder: Directory with all the images.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of the torch tensors related to the image.
            target_dtype: Data type of the torch tensors related to the target.
            channel_order: Order of channel, candidates are `bgr` and `rgb`.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            test_mode: If True, only image without labels will be returned.
        """
        self.data_folder = Path(data_folder)
        self.path = self.data_folder / self.base_folder

        if download:
            self._download()

        if not self.path.is_dir():
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

        csv_path = self.train_csv if train else self.valid_csv

        super().__init__(
            data_folder=self.path,
            csv_path=csv_path,
            transform=transform,
            augment=augment,
            input_dtype=input_dtype,
            target_column='mask',
            target_dtype=target_dtype,
            channel_order=channel_order,
            grayscale=grayscale,
            test_mode=test_mode,
        )

    def _download(self) -> None:
        """Download archive by url to specific folder."""
        if self.path.is_dir():
            print('Files already downloaded and verified')
        else:
            download_and_extract_archive(self.url, self.data_folder.as_posix(),
                                         filename=self.filename, md5=self.tgz_md5)
