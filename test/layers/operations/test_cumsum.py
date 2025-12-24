import torch
from test.utils import convert_and_test
import numpy as np
import pytest


class TorchCumSum(torch.nn.Module):
    def __init__(self, axis):
        super(TorchCumSum, self).__init__()
        self.axis = axis

    def forward(self, x):
        return torch.cumsum(x, dim=self.axis)


@pytest.mark.parametrize('axis', [-1, 1])
def test_cumsum(axis):
    pt_model = TorchCumSum(axis=axis)
    error = convert_and_test(pt_model, (np.random.random((1, 8, 3))), verbose=False)

