import numpy as np
import torch.nn as nn
import pytest

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self, inp, out, kernel_size=3, padding=1, stride=1, bias=False):
        super(LayerTest, self).__init__()
        self.conv = nn.ConvTranspose2d(inp, out, kernel_size=kernel_size, padding=padding,
                                       stride=stride, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('kernel_size', [1, 3, 5])
@pytest.mark.parametrize('padding', [0, 1, 3])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('bias', [True, False])
def test_convtranspose2d(change_ordering, kernel_size, padding, stride, bias):
    outs = np.random.choice([1, 3, 7])
    model = LayerTest(3, outs, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
