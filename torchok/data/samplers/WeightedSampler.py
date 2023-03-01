from torch.utils.data import WeightedRandomSampler
from torchok.constructor import SAMPLERS
from pathlib import Path
import pandas as pd


@SAMPLERS.register_class
class WeightedSampler(WeightedRandomSampler):
    """Init WeightedSampler.

        Args:
            data_folder: Directory with csv file.
            csv_path: Path to the csv file with path to images and annotations.
            weight_column: column name containing weights for each image.
    """
    def __init__(self,
                 data_folder: str,
                 csv_path: str,
                 num_samples: int,
                 weight_column: str = "weight",
                 replacement: bool = True) -> WeightedRandomSampler:

        csv_path = Path(data_folder, csv_path)
        self.weight_column = weight_column
        self.csv = pd.read_csv(csv_path)

        sampler_weights = self.get_sampler_weights()
        super(WeightedSampler, self).__init__(weights=sampler_weights,
                                              num_samples=num_samples,
                                              replacement=replacement)

    def get_sampler_weights(self):
        if self.weight_column not in self.csv.columns:
            raise KeyError(f"Weight column {self.weight_column} doesn't exsist in the csv file")
        return self.csv[self.weight_column].values
