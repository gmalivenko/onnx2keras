import numpy as np
import torch
import torch.nn as nn
import pytest

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self):
        super(LayerTest, self).__init__()

    def forward(self, x):
        return torch.split(x, 224//4, 3)


@pytest.mark.parametrize('change_ordering', [True, False])
def test_split(change_ordering):
    model = LayerTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
