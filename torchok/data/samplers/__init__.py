from torch.utils.data import WeightedRandomSampler
from torchok.constructor import SAMPLERS

SAMPLERS.register_class(WeightedRandomSampler)
