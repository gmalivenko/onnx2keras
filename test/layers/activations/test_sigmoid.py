import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerSigmoid(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerSigmoid, self).__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.sig(x)
        return x


class FSigmoid(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FSigmoid, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.sigmoid(x)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_sigmoid(change_ordering):
    model = LayerSigmoid()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_sigmoid(change_ordering):
    model = FSigmoid()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
