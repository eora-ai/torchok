from typing import List
import torch

from dataclasses import dataclass


@dataclass
class FakeData:
    name: str
    shape: List[int]
    num_repeats: int


class FakeDataGenerator:
    def __init__(self, fake_data_list: List[FakeData], bs: int = 4):
        self.fake_data_list = fake_data_list
        self.bs = bs

    def _generate_fake_data(self, fake_data: FakeData):
        return torch.rand([self.bs] + fake_data.shape)

    def __len__(self):
        list_repeats = [fake_data.num_repeats for fake_data in self.fake_data_list]
        return max(list_repeats)

    def __getitem__(self, idx):
        curr_dict = {}
        for fake_data in self.fake_data_list:
            if idx < fake_data.num_repeats:
                curr_dict[fake_data.name] = self._generate_fake_data(fake_data)
        return curr_dict
