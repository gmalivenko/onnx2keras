import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerSELU(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerSELU, self).__init__()
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.selu(x)
        return x


class FSELU(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FSELU, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.selu(x)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_selu(change_ordering):
    model = LayerSELU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_selu(change_ordering):
    model = FSELU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
