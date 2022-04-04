from typing import List
import torch

from dataclasses import dataclass


@dataclass
class FakeData:
    name: str
    shape: List[int]


class FakeDataGenerator:
    def __init__(self, fake_data_list: List[FakeData], fake_data_sizes: List[int], bs: int = 4):
        self.fake_data_list = fake_data_list
        self.fake_data_sizes = fake_data_sizes
        self.bs = bs
        
    def _generate_fake_data(self, fake_data: FakeData):
        return torch.rand([self.bs] + fake_data.shape)
    
    def __len__(self):
        return max(self.fake_data_sizes)
    
    def __getitem__(self, idx):
        curr_dict = {}
        for count, fake_data in zip(self.fake_data_sizes, self.fake_data_list):
            if idx < count:
                curr_dict[fake_data.name] = self._generate_fake_data(fake_data)
        return curr_dict
    