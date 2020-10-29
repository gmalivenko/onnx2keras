import numpy as np
import torch.nn as nn
import random
import pytest

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self, out, eps, momentum):
        super(LayerTest, self).__init__()
        self.bn = nn.BatchNorm2d(out, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.bn(x)
        return x


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_bn2d(change_ordering):
    inp_size = np.random.randint(10, 100)

    model = LayerTest(inp_size, random.random(), random.random())
    model.eval()

    input_np = np.random.uniform(0, 1, (1, inp_size, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
