from collections import defaultdict
from typing import Any, List, Dict

import torch
from torch.utils.data._utils.collate import default_collate

from torchok.constructor import DATASETS
from torchok.data.datasets.classification.classification import ImageClassificationDataset


@DATASETS.register_class
class SwaVDataset(ImageClassificationDataset):

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        samples = self.get_raw(idx)
        new_samples = []
        for i in range(len(samples['image'])):
            view = {k: v[i] for k, v in samples.items()}
            view = self._apply_transform(self.transform, view)
            view['image'] = view['image'].type(torch.__dict__[self.input_dtype])
            new_samples.append(view)

        return new_samples

    def collate_fn(self, batch: List[List[Dict[str, Any]]]) -> Dict[str, List[torch.Tensor]]:
        num_views = len(batch[0])
        new_batch = defaultdict(list)

        for j in range(num_views):
            cross_batch_view = default_collate([sample_views[j] for sample_views in batch])
            for k, v in cross_batch_view.items():
                new_batch[k].append(v)

        return new_batch
