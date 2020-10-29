import numpy as np
import pytest
import tensorflow as tf

from test.utils import convert_and_test
from torchvision.models import squeezenet1_0, squeezenet1_1


@pytest.mark.slow
@pytest.mark.parametrize('change_ordering', [True, False])
@pytest.mark.parametrize('model_class', [squeezenet1_1, squeezenet1_0])
def test_squeezenet(change_ordering, model_class):
    if not tf.test.gpu_device_name() and not change_ordering:
        pytest.skip("Skip! Since tensorflow Conv2D op currently only supports the NHWC tensor format on the CPU")
    model = model_class()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, change_ordering=change_ordering)
