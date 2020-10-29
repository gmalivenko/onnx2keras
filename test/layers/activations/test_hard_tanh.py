import torch.nn as nn
import random
import pytest
import numpy as np

from test.utils import convert_and_test


class LayerHardtanh(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerHardtanh, self).__init__()
        self.min_val = random.random()
        self.max_val = self.min_val + random.random()
        self.htanh = nn.Hardtanh(min_val=self.min_val, max_val=self.max_val)

    def forward(self, x):
        x = self.htanh(x)
        return x


class FHardtanh(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FHardtanh, self).__init__()
        self.min_val = random.random()
        self.max_val = self.min_val + random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.hardtanh(x, min_val=self.min_val, max_val=self.max_val)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_hardtanh(change_ordering):
    model = LayerHardtanh()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_hardtanh(change_ordering):
    model = LayerHardtanh()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
