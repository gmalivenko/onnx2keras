import numpy as np
import pytest
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2

from test.utils import convert_and_test


@pytest.mark.parametrize('model_class', [resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2,
                                         wide_resnet101_2])
def test_resnet18(model_class):
    model = model_class()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
