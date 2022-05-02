import unittest
import os
import torch
import math
from typing import *
from src.losses import LOSSES

loss_class = LOSSES.get('AngularLoss')
loss_function = loss_class()

class TestPMLLosses(unittest.TestCase):

    def test_pml_loss(self):
        emd = torch.zeros(8, 256)
        label = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        assert math.isclose(1.6094, float(loss_function(emd, label)), rel_tol=1e-04)