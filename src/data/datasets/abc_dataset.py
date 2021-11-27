from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class ABCDataset(Dataset, ABC):
    """An abstract dataset for image classification task"""

    def __init__(self, transform, augment=None, bbox_augment=None):
        self.transform = transform
        self.augment = augment
        self.bbox_augment = bbox_augment
        self.transform_targets = {}

    def update_transform_targets(self, transform_targets):
        self.transform_targets = transform_targets
        self.transform.additional_targets = transform_targets
        self.transform.add_targets(transform_targets)
        if self.augment is not None:
            self.augment.additional_targets = transform_targets
            self.augment.add_targets(transform_targets)

    def apply_transform(self, transform, sample):
        if transform is None:
            return sample

        new_sample = {}
        for source, target in self.transform_targets.items():
            if source in sample:
                if source == 'input' or source == 'target':
                    new_sample[target] = sample[source]
                else:
                    new_sample[source] = sample[source]
        new_sample = transform(**new_sample)
        for source, target in self.transform_targets.items():
            if target in new_sample and (source == 'input' or source == 'target'):
                sample[source] = new_sample[target]
            elif source in new_sample:
                sample[source] = new_sample[source]
        return sample

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def get_raw(self, item):
        pass
