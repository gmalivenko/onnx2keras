import torch.nn as nn
import torch
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerPReLU(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self, num_params=3):
        super(LayerPReLU, self).__init__()
        self.num_params = num_params
        self.prelu = nn.PReLU(num_params)

    def forward(self, x):
        x = self.prelu(x)
        return x


class FPReLU(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self, num_params=3):
        super(FPReLU, self).__init__()
        self.num_params = num_params

    def forward(self, x):
        from torch.nn import functional as F
        weights = torch.FloatTensor(torch.rand(self.num_params).numpy())
        return F.prelu(x, weight=weights)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_prelu(change_ordering):
    model = LayerPReLU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_prelu(change_ordering):
    model = FPReLU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
