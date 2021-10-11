from random import sample as sample_fn, choice

import pandas as pd
import torch
from torch.utils.data.dataloader import default_collate

from src.registry import DATASETS
from .classification import ImageDataset


@DATASETS.register_class
class SamplingDataset(ImageDataset):
    """Dataset that combine several random samples of the same class into one batch"""
    def __init__(self, max_group_size: int, adjust_small_groups: bool = False, one_label_per_group: bool = False,
                 group_by: str = None, **dataset_params):
        """
        Args:
            data_folder: Directory with all the images.
            path_to_datalist: Path to the csv file with path to images and annotations.
                Path to images must be under column `image_path` and annotations must be under `label` column
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            input_dtype: Data type of of the torch tensors related to the image.
            target_dtype: Data type of of the torch tensors related to the target.
            target_column: Name of the column that contains target labels.
            grayscale: If True, image will be read as grayscale otherwise as RGB.
            expand_rate: A multiplier that shows how many times the dataset will be larger than its real size.
                Useful for small datasets.
            test_mode: If True, only image without labels will be returned.
            max_group_size: A maximum number of the images of the same class in the batch.
            adjust_small_groups: If True, resample images to be same amount as `max_group_size`.
            one_label_per_group: If True, batch will contain input of shape [B*N, C, H, W] and target of shape [B].
        """
        super().__init__(**dataset_params)
        self.group_by = group_by or self.target_column
        self.max_group_size = max_group_size
        self.adjust_small_groups = adjust_small_groups
        self.one_label_per_group = one_label_per_group
        self.groups = pd.Series(self.csv.groupby(self.group_by).groups).reset_index(drop=True)

    def __getitem__(self, idx: int):
        group = tuple(self.groups[idx])
        if len(group) < self.max_group_size:
            selected = list(group)
            if self.adjust_small_groups:
                while len(selected) < self.max_group_size:
                    selected.append(choice(group))
        else:
            selected = sample_fn(group, self.max_group_size)
        samples = [super(SamplingDataset, self).__getitem__(i) for i in selected]
        if self.one_label_per_group:
            if not self.adjust_small_groups:
                raise ValueError('`adjust_small_groups` must be True when `one_label_per_group` is True')
            images = torch.stack([sample['input'] for sample in samples], 0)
            index = torch.tensor([sample['index'] for sample in samples])
            data = {'input': images, 'index': index}
            if not self.test_mode:
                data['target'] = samples[0]['target']
            return data
        return samples

    def __len__(self):
        return len(self.groups)

    def collate_fn(self, batch):
        if self.one_label_per_group:
            batch = default_collate(batch)
            data = batch['input']
            b, n, c, h, w = data.shape
            batch['input'] = data.view(b * n, c, h, w)
        else:
            batch = [elem for group in batch for elem in group]
            batch = default_collate(batch)
        return batch


@DATASETS.register_class
class DeterminedSamplingDataset(SamplingDataset):
    """Dataset that combine several first samples of the same class into one batch"""
    def __getitem__(self, idx: int):
        group = tuple(self.groups[idx])
        if len(group) < self.max_group_size:
            selected = list(group)
            if self.adjust_small_groups:
                while len(selected) < self.max_group_size:
                    selected.extend(list(group))
            selected = selected[:self.max_group_size]
        else:
            selected = group[:self.max_group_size]
        samples = [super(SamplingDataset, self).__getitem__(i) for i in selected]
        if self.one_label_per_group:
            if not self.adjust_small_groups:
                raise ValueError('`adjust_small_groups` must be True when `one_label_per_group` is True')
            images = torch.stack([sample['input'] for sample in samples], 0)
            index = torch.tensor([sample['index'] for sample in samples])
            data = {'input': images, 'index': index}
            if not self.test_mode:
                data['target'] = samples[0]['target']
            return data
        return samples
