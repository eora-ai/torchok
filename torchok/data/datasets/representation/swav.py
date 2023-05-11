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
            view = {name: views[i] for name, views in samples.items()}
            view = self._apply_transform(self.transform, view)
            view['image'] = view['image'].type(torch.__dict__[self.input_dtype])
            new_samples.append(view)

        return new_samples

    def collate_fn(self, batch: List[List[Dict[str, Any]]]) -> Dict[str, List[torch.Tensor]]:
        """
        Construct batch from multiview samples

        Args:
            batch: list of samples where each sample is a list of several views from the same image.
                Example: sample_i = [view_1_i, ..., view_j_i,  ..., view_m_i], where
                         m - number of views, view_j_i = {'image': crop_j_i, ...} and
                         crop_j_i is a j-th crop from i-th image in the dataset.

        Returns:
            Dict where values are list of batches stacked along the view.
            Example with the same notation as in Args:
                batch = {'image': [batch_view_1, ..., batch_view_j, ..., batch_view_m], ...}, where
                batch_view_j = torch.stack([view_j_1, view_j_2, ..., view_j_i, ..., view_j_n]) and
                n - batch size.
        """

        num_views = len(batch[0])
        new_batch = defaultdict(list)

        for j in range(num_views):
            cross_batch_view = default_collate([sample_views[j] for sample_views in batch])
            for k, v in cross_batch_view.items():
                new_batch[k].append(v)

        return new_batch
