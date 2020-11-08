import numpy as np
import pytest
import tensorflow as tf

from test.utils import convert_and_test
from torchvision.models import mobilenet_v2


@pytest.mark.parametrize('change_ordering', [True, False])
def test_mobilenetv2(change_ordering):
    if not tf.test.gpu_device_name() and not change_ordering:
        pytest.skip("Skip! Since tensorflow Conv2D op currently only supports the NHWC tensor format on the CPU")
    model = mobilenet_v2()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
