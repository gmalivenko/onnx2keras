import torch.nn as nn
import random
import pytest
import numpy as np

from test.utils import convert_and_test


class LayerLeakyReLU(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerLeakyReLU, self).__init__()
        self.negative_slope = random.random()
        self.leaky_relu = nn.LeakyReLU(negative_slope=self.negative_slope)

    def forward(self, x):
        x = self.leaky_relu(x)
        return x


class FLeakyReLU(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FLeakyReLU, self).__init__()
        self.negative_slope = random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.leaky_relu(x, self.negative_slope)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_leaky_relu(change_ordering):
    model = LayerLeakyReLU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_leaky_relu(change_ordering):
    model = FLeakyReLU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
