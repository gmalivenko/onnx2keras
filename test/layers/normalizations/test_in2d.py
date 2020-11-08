import numpy as np
import torch.nn as nn
import random
import pytest

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self, out, eps, momentum):
        super(LayerTest, self).__init__()
        self.in2d = nn.InstanceNorm2d(out, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.in2d(x)
        return x


@pytest.mark.repeat(10)
# sometimes error is a little bit greater than 1e-5
# maybe it can be problem described here
# https://discuss.pytorch.org/t/instance-norm-implement-by-basic-operations-has-different-result-comparing-to-torch-nn-instancenorm2d/87470/2
@pytest.mark.parametrize('epsilon', [1e-4])
@pytest.mark.parametrize('change_ordering', [True, False])
def test_instancenorm(change_ordering, epsilon):
    inp_size = np.random.randint(10, 100)

    model = LayerTest(inp_size, random.random(), random.random())
    model.eval()

    input_np = np.random.uniform(0, 1, (1, inp_size, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering, epsilon=1e-4)
