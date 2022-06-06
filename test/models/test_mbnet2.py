import numpy as np
import pytest
from torchvision.models import mobilenet_v2

from test.utils import convert_and_test


@pytest.mark.parametrize('pretrained', [True, False])
def test_mobilenetv2(pretrained):
    model = mobilenet_v2(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
