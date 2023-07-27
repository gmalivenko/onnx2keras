import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytest

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self, inp, out, bias=False):
        super(LayerTest, self).__init__()
        self.fc = nn.Linear(inp, out, bias=bias)

    def forward(self, x):
        x = self.fc(x)
        return x


class DetTest(nn.Module):
    def __init__(self):
        super(DetTest, self).__init__()

    def forward(self, x):
        return torch.det(x)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('bias', [True, False])
def test_linear(change_ordering, bias):
    ins = np.random.choice([1, 3, 7, 128])

    model = LayerTest(ins, np.random.choice([1, 3, 7, 128]), bias=bias)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, ins))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


def test_det():
    matrix = np.random.random((2, 4, 4))
    error = convert_and_test(DetTest(), matrix, verbose=False, change_ordering=False)
