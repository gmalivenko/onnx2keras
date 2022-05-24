import numpy as np
import pytest
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [mobilenet_v3_small, mobilenet_v3_large])
def test_mobilenetv3(model_class):
    model = model_class()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 124, 124))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
