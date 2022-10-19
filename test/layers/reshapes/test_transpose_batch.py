import numpy as np
import torch
import torch.nn as nn
import pytest

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self):
        super(LayerTest, self).__init__()

    def forward(self, x):
        x = torch.permute(x, (1, 0, 2, 3))
        x = torch.permute(x, (1, 0, 2, 3))
        return x


@pytest.mark.parametrize('change_ordering', [True])
def test_transpose_batch_and_abs(change_ordering):
    model = LayerTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 28, 28))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
