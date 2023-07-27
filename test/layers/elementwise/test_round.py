import torch
import pytest
from test.utils import convert_and_test
import numpy as np


class TorchRound(torch.nn.Module):
    def __init__(self):
        super(TorchRound, self).__init__()

    def forward(self, x):
        return torch.round(x)


def test_round():
    pt_model = TorchRound()
    error = convert_and_test(pt_model, (np.random.random((1, 8, 3))), verbose=False)

