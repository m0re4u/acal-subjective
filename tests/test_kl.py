import unittest

import numpy as np
import torch
import torch.nn.functional as F
import torch.testing

from annotator_diversity.soft_trainer import torch_kl


class TestKLDivSanity(unittest.TestCase):
    def test_stackoverflow(self):
        """
        :)
        https://stackoverflow.com/questions/49886369/kl-divergence-for-two-probability-distributions-in-pytorch
        """
        self.P = torch.tensor([0.36, 0.48, 0.16])
        self.Q = torch.tensor([0.333, 0.333, 0.333])
        manual_outcome = (self.P * (self.P / self.Q).log()).sum()
        torch_outcome = F.kl_div(self.Q.log(), self.P, None, None, "sum")
        torch.testing.assert_close(manual_outcome, torch.tensor(0.0863))
        torch.testing.assert_close(torch_outcome, torch.tensor(0.0863))

    def test_torch_loss(self):
        self.P = torch.tensor([0.36, 0.48, 0.16])
        self.Q = torch.tensor([0.333, 0.333, 0.333])
        torch_F_outcome = F.kl_div(self.Q.log(), self.P, None, None, "sum")
        self.kldivloss = torch.nn.KLDivLoss(reduction="sum")
        torch_nn_outcome = self.kldivloss(self.Q.log(), self.P)
        torch.testing.assert_close(torch_F_outcome, torch_nn_outcome)

    def test_torch_loss_onehot(self):
        self.P = torch.tensor([0.36, 0.48, 0.16])
        self.Q = torch.tensor([0, 0, 1.0])
        torch_F_outcome = F.kl_div(self.Q.log(), self.P, None, None, "sum")
        outcome = torch.tensor(torch.inf)
        self.kldivloss = torch.nn.KLDivLoss(reduction="sum")
        torch_nn_outcome = self.kldivloss(self.Q.log(), self.P)
        torch.testing.assert_close(torch_F_outcome, torch_nn_outcome)
        torch.testing.assert_close(torch_F_outcome, outcome)
        self.P = F.softmax(self.P, dim=0)
        self.Q = F.softmax(self.Q, dim=0)
        torch_F_outcome = F.kl_div(self.Q.log(), self.P, None, None, "sum")
        outcome = torch.tensor(0.1834)
        torch_nn_outcome = self.kldivloss(self.Q.log(), self.P)
        torch.testing.assert_close(
            torch_F_outcome, torch_nn_outcome, atol=1.3e-04, rtol=3e-05
        )
        torch.testing.assert_close(torch_F_outcome, outcome, atol=1.3e-04, rtol=3e-05)


class TestKLDivImplementation(unittest.TestCase):
    def setUp(self):
        self.P = np.array([[0.36, 0.48, 0.16]])
        self.Q = np.array([[0.333, 0.333, 0.333]])
        self.P = torch.from_numpy(self.P)
        self.Q = torch.from_numpy(self.Q)

    def test_torch(self):
        # Swapped P and Q order w.r.t. above, since PyTorch has an implementation that swaps P and
        # Q internally.
        #
        # See the discussion here: https://github.com/pytorch/pytorch/issues/57459
        # and the note on the docs: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        torch_outcome = torch_kl(self.Q, self.P)
        torch.testing.assert_close(
            torch_outcome, torch.tensor(0.0863, dtype=torch.float64)
        )
