import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerReLU(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(x)
        return x


class FReLU(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FReLU, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.relu(x)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_relu(change_ordering):
    model = LayerReLU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_relu(change_ordering):
    model = FReLU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
