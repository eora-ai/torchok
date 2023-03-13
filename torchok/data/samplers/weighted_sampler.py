from pathlib import Path

import pandas as pd
from torch.utils.data import WeightedRandomSampler

from torchok.constructor import SAMPLERS


@SAMPLERS.register_class
class WeightedSampler(WeightedRandomSampler):
    """Init WeightedSampler.

        Args:
            data_folder: Directory with csv file.
            annotation_path: Path to the csv file with path to images and annotations.
            weight_column: column name containing weights for each image.
    """

    def __init__(self,
                 data_folder: str,
                 annotation_path: str,
                 num_samples: int,
                 weight_column: str = "weight",
                 replacement: bool = True
                 ):

        self.data_folder = Path(data_folder)
        self.weight_column = weight_column

        if annotation_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_folder / annotation_path)
        elif annotation_path.endswith('.pkl'):
            self.df = pd.read_pickle(self.data_folder / annotation_path)
        else:
            raise ValueError('Detection dataset error. Annotation path is not in `csv` or `pkl` format')

        sampler_weights = self.get_sampler_weights()
        super(WeightedSampler, self).__init__(weights=sampler_weights,
                                              num_samples=num_samples,
                                              replacement=replacement)

    def get_sampler_weights(self):
        if self.weight_column not in self.df.columns:
            raise KeyError(f"Weight column {self.weight_column} doesn't exist in the csv file")
        return self.df[self.weight_column].values
