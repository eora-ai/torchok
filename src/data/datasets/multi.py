import math
import random
from collections import defaultdict
from fractions import Fraction as Fr

from albumentations import BasicTransform
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from src.registry import DATASETS


def random_range(start, stop=None, step=None):
    """
    Creates an random iterator over given range.
    Equivalent for iter(shuffle(list(range(start, stop, step)))) but has O(1) for memory.
    """
    # Set a default values the same way "range" does.
    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1
    # Compute the number of numbers in this range.
    maximum = (stop - start) // step
    # Seed range with a random integer.
    value = random.randint(0, maximum)
    #
    # Construct an offset, multiplier, and modulus for a linear
    # congruential generator. These generators are cyclic and
    # non-repeating when they maintain the properties:
    #
    #   1) "modulus" and "offset" are relatively prime.
    #   2) ["multiplier" - 1] is divisible by all prime factors of "modulus".
    #   3) ["multiplier" - 1] is divisible by 4 if "modulus" is divisible by 4.
    #
    offset = random.randint(0, maximum) * 2 + 1  # Pick a random odd-valued offset.
    multiplier = 4 * (maximum // 4) + 1  # Pick a multiplier 1 greater than a multiple of 4.
    # Pick a modulus just big enough to generate all numbers (power of 2).
    modulus = int(2 ** math.ceil(math.log2(maximum)))
    # Track how many random numbers have been returned.
    found = 0
    while found < maximum:
        # If this is a valid value, yield it in generator fashion.
        if value < maximum:
            found += 1
            # Use a mapping to convert a standard range into the desired range.
            yield (value * step) + start
        # Calculate the next value in the sequence.
        value = (value * multiplier + offset) % modulus


@DATASETS.register_class
class MultiDataset(Dataset):
    def __init__(self, datasets: list, same_transforms: bool, transform: BasicTransform,
                 augment: BasicTransform = None, align_by: str = None):
        """
        :param datasets: List of dicts where each dict contains information about nested dataset. Each must contain the
                following fields:
                    - type: str, class name of the Dataset
                    - name: str, name that will be used in batch collection to identify certain dataset
                    - scale (optional): int, in case of alignment this number will be used to adapt number of samples
                    in the epoch.
                        Example: dataset A has `n` samples, scale=a and dataset B has `m` samples, scale=b. If
                        `align_by` is None then the multi-dataset will have size of `n+m`. If `align_by` is `A` then
                        the multi-dataset will have size `n+floor(n*b/a)`. If `align_by` is `B` then the
                        multi-dataset will have size `m+floor(m*a/b)`. Default 1.
                    - params: list of the parameters for the dataset except transform and augment.
        :param same_transforms: If True set `transform` and `augment` to all nested datasets
                otherwise takes inner indexed transformations from `transform` and `augment`.
        :param transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
        :param augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
        :param align_by: name of the dataset whose length will be used as a indicator of total dataset length. All
                samples from that dataset will be shown during the epoch meanwhile samples from other datasets will be
                selected randomly. In case of smaller dataset some samples may be shown several times during the epoch.
                In case of bigger dataset some samples may be not shown times during the epoch. Randomness over several
                epochs neutralize negative effect.

        MultiDataset provides batch_sampler that forms batch according to the scales. It may be used only with the
        datasets alignment.
        """
        self.align_by = align_by
        self.align = self.align_by is not None
        self.datasets = {}

        if self.align and not any([align_by == d['name'] for d in datasets]):
            raise ValueError('`align_by` is the name of the dataset and must present in the dataset list')

        scales = {}

        for i, dataset_dict in enumerate(datasets):
            name = dataset_dict['name']
            params = dataset_dict['params']
            if same_transforms:
                params['transform'] = transform
                params['augment'] = augment
            else:
                params['transform'] = transform.transforms[i]
                params['augment'] = augment.transforms[i] if augment else None

            scale = dataset_dict.get('scale', 1)
            if scale < 1 or not isinstance(scale, int):
                raise ValueError('`scale` of the dataset must be natural number')
            scale = Fr(scale, 1)
            scales[name] = scale

            dataset_type = DATASETS.get(dataset_dict['type'])
            dataset = dataset_type(**params)
            if name == self.align_by:
                self.base_dataset_len = len(dataset) / scale

            self.datasets[name] = dataset

        if self.align:
            self.scales = scales

    def __len__(self):
        if self.align:
            return math.floor(self.base_dataset_len * sum(self.scales.values()))
        else:
            return sum(map(len, self.datasets.values()))

    def __getitem__(self, idx: int):
        for name, dataset in self.datasets.items():
            length = math.floor(self.base_dataset_len * self.scales[name]) if self.align else len(dataset)
            if length <= idx:
                idx -= length
                continue

            if self.align and name != self.align_by:
                idx = random.randrange(len(dataset))
            sample = dataset[idx]
            sample['name'] = name
            return sample

    def collate_fn(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        new_batch = defaultdict(list)
        for i, elem in enumerate(batch):
            sample = {}
            name = elem.pop('name')
            for k in list(elem.keys()):
                sample[f'{k}_{name}'] = elem.pop(k)
            new_batch[name].append(sample)
        output = {}
        for data in new_batch.values():
            output.update(default_collate(data))

        return output

    def batch_sampler(self, batch_size, shuffle=False, drop_last=False):
        return FairMultiDatasetBatchSampler(self, batch_size, drop_last, shuffle)


class FairMultiDatasetBatchSampler:
    r"""Wraps MultiDataset to yield a mini-batch of indices from different datasets in certain proportion.

    Args:
        dataset (MultiDataset): an instance of MultiDataset in align mode
        batch_size (int): Size of mini-batch.
        shuffle (bool): if ``True``, indices of aligned dataset will be shuffled.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    def __init__(self, dataset, batch_size, drop_last, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        min_num_samples = sum(self.dataset.scales.values())
        if self.batch_size % min_num_samples != 0:
            raise ValueError('`batch_size` must be divisible by sum of all dataset scales')
        batch_multiplier = self.batch_size // min_num_samples

        indices = {}
        last_idx = 0
        for name, dataset in self.dataset.datasets.items():
            dataset_size = math.floor(self.dataset.scales[name] * self.dataset.base_dataset_len)
            if name != self.dataset.align_by or not self.shuffle:
                indices[name] = iter(range(last_idx, last_idx + dataset_size))
            else:
                indices[name] = random_range(last_idx, last_idx + dataset_size)
            last_idx += dataset_size

        is_over = False
        while not is_over:
            batch_idx = []
            for n, scale in self.dataset.scales.items():
                for _ in range(int(scale * batch_multiplier)):
                    try:
                        batch_idx.append(next(indices[n]))
                    except StopIteration:
                        is_over = True
                        break
            if len(batch_idx) == self.batch_size or (not self.drop_last and len(batch_idx) > 0):
                yield batch_idx

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size  # type: ignore
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # type: ignore


@DATASETS.register_class
class DomainAdaptationDataset(Dataset):
    """A generic dataset for images."""

    def __init__(self, source_dataset: dict, target_dataset: dict, transform: BasicTransform,
                 augment: BasicTransform = None, same_transforms: bool = True, balance_by: str = 'none'):
        """
        Args:
            source_dataset: Dict contains information about source dataset.
            target_dataset: Dict contains information about target dataset.
            same_transforms: If True set `transform` and `augment` to all nested datasets
                otherwise takes inner indexed transformations from `transform` and `augment`.
            transform: Transform to be applied on a sample. This should have the
                interface of transforms in `albumentations` library.
            augment: Optional augment to be applied on a sample.
                This should have the interface of transforms in `albumentations` library.
            balance_by: Balancing source or target dataset by changing amount of returned samples in selected dataset
                with respect to the size of opposite dataset.
                Can be `none`|`source`|`target`. If `none` than there will no any a balancing otherwise selected
                dataset will be balanced.
        """
        if balance_by not in ['none', 'source', 'target']:
            raise ValueError('`balance_by` must be `none`|`source`|`target`')
        self.balance_by = balance_by
        datasets = []
        for i, dataset in enumerate([source_dataset, target_dataset]):
            if same_transforms or augment is None:
                loc_augment = augment
            else:
                loc_augment = augment.transforms[i]

            dataset_type = DATASETS.get(dataset['type'])
            datasets.append(dataset_type(transform=transform,
                                         augment=loc_augment,
                                         **dataset['params']))
        self.source_dataset, self.target_dataset = datasets

        self.target_size = len(self.target_dataset)
        self.source_size = len(self.source_dataset)

    def __len__(self):
        if self.balance_by == 'none':
            return self.source_size + self.target_size
        elif self.balance_by == 'source':
            return 2 * self.target_size
        elif self.balance_by == 'target':
            return 2 * self.source_size
        else:
            raise ValueError('`balance_by` must be `none`|`source`|`target`')

    def __getitem__(self, idx: int):
        new_sample = {}
        if self.balance_by == 'none':
            new_sample['is_target'] = self.source_size <= idx
            if new_sample['is_target']:
                idx -= len(self.source_size)
        elif self.balance_by == 'source':
            new_sample['is_target'] = self.target_size <= idx
            if self.target_size <= idx:
                idx -= len(self.target_size)
            else:
                idx = random.randint(0, self.source_size - 1)
        elif self.balance_by == 'target':
            new_sample['is_target'] = self.source_size <= idx
            if new_sample['is_target']:
                idx = random.randint(0, self.target_size - 1)
        else:
            raise ValueError('`balance_by` must be `none`|`source`|`target`')

        if new_sample['is_target']:
            sample = self.target_dataset[idx]
            new_sample['input'] = sample['input']
            if 'target' in sample:
                new_sample['dest_target'] = sample['target']
        else:
            new_sample.update(self.source_dataset[idx])
        return new_sample

    @staticmethod
    def collate_fn(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        new_batch = defaultdict(list)
        for i, elem in enumerate(batch):
            is_target = elem.pop('is_target')
            new_batch['inv_index' if is_target else 'index'].append(i)

            for field, value in elem.items():
                new_batch[field].append(value)
        for name, target in new_batch.items():
            new_batch[name] = default_collate(target)
        return dict(new_batch)
