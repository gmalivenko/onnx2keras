import torch.nn as nn
import pytest
import numpy as np

from test.utils import convert_and_test


class LayerLogSigmoid(nn.Module):
    """
    Test for nn.layers based types
    """
    def __init__(self):
        super(LayerLogSigmoid, self).__init__()
        self.sig = nn.LogSigmoid()

    def forward(self, x):
        x = self.sig(x)
        return x


class FLogSigmoid(nn.Module):
    """
    Test for nn.functional types
    """
    def __init__(self):
        super(FLogSigmoid, self).__init__()

    def forward(self, x):
        from torch.nn import functional as F
        return F.logsigmoid(x)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_layer_logsigmoid(change_ordering):
    model = LayerLogSigmoid()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_f_logsigmoid(change_ordering):
    model = FLogSigmoid()
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
