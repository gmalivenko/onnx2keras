import torch
from test.utils import convert_and_test
import numpy as np
import pytest


class TorchReduceProd(torch.nn.Module):
    def __init__(self, axis, keepdim):
        super(TorchReduceProd, self).__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, x):
        return torch.prod(x, dim=self.axis, keepdim=self.keepdim)


class TorchReduceMin(torch.nn.Module):
    def __init__(self, axis, keepdim):
        super(TorchReduceMin, self).__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, x):
        return torch.min(x, dim=self.axis, keepdim=self.keepdim)


@pytest.mark.parametrize('axis', [-1, 1])
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_prod(axis, keepdim):
    pt_model = TorchReduceProd(axis=axis, keepdim=keepdim)
    error = convert_and_test(pt_model, (np.random.random((1, 8, 3))), verbose=False)


@pytest.mark.parametrize('axis', [-1, 1])
@pytest.mark.parametrize('keepdim', [True, False])
def test_reduce_min(axis, keepdim):
    pt_model = TorchReduceMin(axis=axis, keepdim=keepdim)
    error = convert_and_test(pt_model, (np.random.random((1, 8, 3))), verbose=False)

