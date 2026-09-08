import numpy as np
import torch.nn as nn
import pytest

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self, perm):
        super(LayerTest, self).__init__()
        self.perm = perm

    def forward(self, x):
        x = x.permute(*self.perm)
        return x


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('perm', [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3),
                                  (0, 2, 3, 1), (0, 3, 1, 2), (0, 3, 2, 1)])
def test_transpose(change_ordering, perm):
    model = LayerTest(perm)
    model.eval()

    # The dimensions are all different on purpose, so that a permutation that is not
    # remapped for the target ordering cannot accidentally produce the expected shape.
    input_np = np.random.uniform(0, 1, (1, 3, 16, 32))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
