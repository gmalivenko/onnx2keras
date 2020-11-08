import torch.nn as nn
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerSoftmax(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self, dim):
        super(LayerSoftmax, self).__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        x = self.softmax(x)
        return x


class FSoftmax(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self, dim):
        super(FSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        from torch.nn import functional as F
        return F.softmax(x, self.dim)


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('dim', [0, 1, 2, 3])
def test_layer_softmax(change_ordering, dim):
    model = LayerSoftmax(dim)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('dim', [0, 1, 2, 3])
def test_f_softmax(change_ordering, dim):
    model = FSoftmax(dim)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
