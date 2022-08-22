import torch.nn as nn


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)
        self.linear = nn.Linear(1, 1)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = Block()
        self.block2 = Block()
        self.linear = nn.Linear(1, 1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()
        self.linear = nn.Linear(1, 1)
