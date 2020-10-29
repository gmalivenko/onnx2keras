import numpy as np
import torch.nn as nn
import pytest
import tensorflow as tf

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self):
        super(LayerTest, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(x)
        return x


@pytest.mark.repeat(10)
@pytest.mark.parametrize('change_ordering', [True, False])
def test_global_avgpool2d(change_ordering):
    if not tf.test.gpu_device_name() and not change_ordering:
        pytest.skip("Skip! Since tensorflow AvgPoolingOp op currently only supports the NHWC tensor format on the CPU")
    model = LayerTest()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
