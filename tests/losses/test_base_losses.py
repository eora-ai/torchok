import unittest

import torch
from torch.nn import Module

from src.losses.base import JointLoss

class Loss1(Module):
    def forward(self, input, target):
        return torch.abs(input * 10. - target)


class Loss2(Module):
    def forward(self, input, target):
        return torch.abs(input * 20. - target)


class TestJointLoss(unittest.TestCase):
    def test_weighted_loss_when_two_losses_with_weights_specified(self):
        joint_loss = JointLoss(
            losses=[Loss1(), Loss2()],
            tags=['loss1', 'loss2'],
            mappings=[
                         {
                             'input': 'x',
                             'target': 'y'
                         }
                     ] * 2,
            weights=[0.7, 0.3]
        )

        total_loss, tagged_loss_values = joint_loss.forward(x=torch.ones(1), y=torch.full((1,), fill_value=5.))

        torch.testing.assert_allclose(total_loss, torch.tensor([8.]))

    def test_weighted_loss_when_two_losses_without_weights_specified(self):
        joint_loss = JointLoss(
            losses=[Loss1(), Loss2()],
            tags=['loss1', 'loss2'],
            mappings=[
                         {
                             'input': 'x',
                             'target': 'y'
                         }
                     ] * 2,
            weights=[None, None]
        )

        total_loss, tagged_loss_values = joint_loss.forward(x=torch.ones(1), y=torch.full((1,), fill_value=5.))

        torch.testing.assert_allclose(total_loss, torch.tensor([10.]))

    def test_value_error_when_weights_specified_not_for_each_loss(self):
        joint_loss = JointLoss(
            losses=[Loss1(), Loss2()],
            tags=['loss1', 'loss2'],
            mappings=[
                         {
                             'input': 'x',
                             'target': 'y'
                         }
                     ] * 2,
            weights=[0.7, 0.3]
        )

        total_loss, tagged_loss_values = joint_loss.forward(x=torch.ones(1), y=torch.full((1,), fill_value=5.))

        torch.testing.assert_allclose(total_loss, torch.tensor([8.]))
        torch.testing.assert_allclose(tagged_loss_values['loss1'], torch.tensor([5.]))
        torch.testing.assert_allclose(tagged_loss_values['loss2'], torch.tensor([15.]))

    def test_direct_loss_access_when_two_losses_specified(self):
        with self.assertRaises(ValueError):
            JointLoss(
                losses=[Loss1(), Loss2()],
                tags=['loss1', 'loss2'],
                mappings=[
                             {
                                 'input': 'x',
                                 'target': 'y'
                             }
                         ] * 2,
                weights=[0.7, None]
            )
