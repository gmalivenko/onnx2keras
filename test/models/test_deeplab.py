import numpy as np
import pytest
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large])
def test_deeplab(model_class):
    model = model_class()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 256, 256))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
