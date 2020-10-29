import numpy as np
import torch.nn as nn
import pytest
import tensorflow as tf

from test.utils import convert_and_test


class LayerTest(nn.Module):
    def __init__(self, inp, out, kernel_size=3, padding=1, stride=1, bias=False, dilation=1, groups=1):
        super(LayerTest, self).__init__()
        self.conv = nn.Conv2d(
            inp, out, kernel_size=kernel_size, padding=padding,
            stride=stride, bias=bias, dilation=dilation, groups=groups
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def func(change_ordering, kernel_size, padding, stride, bias, dilation, groups):
    if not tf.test.gpu_device_name() and not change_ordering:
        pytest.skip("Skip! Since tensorflow Conv2D op currently only supports the NHWC tensor format on the CPU")
    if stride > 1 and dilation > 1:
        pytest.skip("strides > 1 not supported in conjunction with dilation_rate > 1")
    model = LayerTest(
        groups * 3, groups,
        kernel_size=kernel_size, padding=padding,
        stride=stride, bias=bias, dilation=dilation, groups=groups)
    model.eval()
    input_np = np.random.uniform(0, 1, (1, groups * 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('kernel_size', [1, 3, 5, 7])
@pytest.mark.parametrize('padding', [0, 1, 3, 5])
@pytest.mark.parametrize('stride', [1])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('dilation', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2, 3])
def test_conv2d_case1(change_ordering, kernel_size, padding, stride, bias, dilation, groups):
    func(change_ordering, kernel_size, padding, stride, bias, dilation, groups)


@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('kernel_size', [1, 3, 5, 7])
@pytest.mark.parametrize('padding', [0, 1, 3, 5])
@pytest.mark.parametrize('stride', [1, 2, 3])
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('dilation', [1])
@pytest.mark.parametrize('groups', [1, 2, 3])
def test_conv2d_case2(change_ordering, kernel_size, padding, stride, bias, dilation, groups):
    func(change_ordering, kernel_size, padding, stride, bias, dilation, groups)
