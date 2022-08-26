from typing import Union, Optional, Dict
import pandas as pd

import torch
from torch import Tensor
from albumentations import BasicTransform
from albumentations.core.composition import BaseCompose
from torchok.data.datasets.examples.sop import SOP
from torchok.constructor import DATASETS


@DATASETS.register_class
class PAIRWISE_SOP(SOP):
    """A class represent Stanford Online Products - SOP dataset.
    
    Additionally, we collected Stanford Online Products dataset: 120k images of 23k classes of online products 
    for metric learning. The homepage of SOP is https://cvgl.stanford.edu/projects/lifted_struct/.
    """
    train_txt = 'Ebay_train_pairwised.txt'
    test_txt = 'Ebay_test_pairwised.txt'

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
        super().__init__(train, download, data_folder, transform, augment, image_dtype, grayscale, test_mode)
        self.anchor_paths_column = 'anchor'
        self.positive_paths_column = 'positive'
        self.negative_paths_column = 'negative'

    def __getitem__(self, idx: int) -> dict:
        """Get item sample.

        Returns:
            output: dict, where
            output['anchor'] - Anchor.
            output['positive'] - Positive.
            output['negative'] - Negative.
            sample['index'] - Index.
        """
        record = self.csv.iloc[idx]

        output = {'anchor': self.__image_preparation(record, self.anchor_paths_column),
                  'positive': self.__image_preparation(record, self.positive_paths_column),
                  'negative': self.__image_preparation(record, self.negative_paths_column),
                  'index': idx}

        return output

    def __image_preparation(self, record: pd.Series, column_name: str) -> Dict[str, Tensor]:
        image = self._read_image(self.path / record[column_name])
        sample = {"image": image}
        sample = self._apply_transform(self.augment, sample)
        sample = self._apply_transform(self.transform, sample)
        image = sample['image'].type(torch.__dict__[self.image_dtype])
        return image
