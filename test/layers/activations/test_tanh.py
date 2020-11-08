import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerTanh(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerTanh, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x)
        return x


class FTanh(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FTanh, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.tanh(x)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_tanh(change_ordering):
    model = LayerTanh()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_tanh(change_ordering):
    model = FTanh()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
