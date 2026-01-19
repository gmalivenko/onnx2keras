import numpy as np
import torch.nn as nn
import pytest
import tensorflow as tf

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self,  kernel_size=3, stride=1):
        super(LayerTest, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, ceil_mode=True)

    def forward(self, x):
        x = self.pool(x)
        return x


@pytest.mark.slow
@pytest.mark.parametrize('change_ordering', [True])
@pytest.mark.parametrize('kernel_size', [3])
@pytest.mark.parametrize('stride', [2])
def test_maxpool3d(change_ordering, kernel_size, stride):
    if not tf.test.gpu_device_name() and not change_ordering:
        pytest.skip("Skip! Since tensorflow MaxPoolingOp op currently only supports the NHWC tensor format on the CPU")
    model = LayerTest(kernel_size=kernel_size, stride=stride)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 19, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
