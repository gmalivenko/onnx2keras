import torch.nn as nn
import random
import numpy as np
import pytest

from test.utils import convert_and_test


class LayerELU(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerELU, self).__init__()
        self.alpha = random.random()
        self.elu = nn.ELU(alpha=self.alpha)

    def forward(self, x):
        x = self.elu(x)
        return x


class FPELU(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FPELU, self).__init__()
        self.alpha = random.random()

    def forward(self, x):
        from torch.nn import functional as F
        return F.elu(x, alpha=self.alpha)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_elu(change_ordering):
    model = LayerELU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_fp_elu(change_ordering):
    model = FPELU()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
