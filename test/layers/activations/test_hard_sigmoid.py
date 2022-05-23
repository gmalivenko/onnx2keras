import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerHardSigmoid(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerHardSigmoid, self).__init__()
        self.hard_sig = nn.Hardsigmoid()

    def forward(self, x):
        x = self.hard_sig(x)
        return x


class FHardSigmoid(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FHardSigmoid, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.hardsigmoid(x)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_sigmoid(change_ordering):
    model = LayerHardSigmoid()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_hard_sigmoid(change_ordering):
    model = FHardSigmoid()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)