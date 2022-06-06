import numpy as np
import pytest
from torchvision.models import resnext50_32x4d, resnext101_32x8d

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [resnext50_32x4d, resnext101_32x8d])
@pytest.mark.parametrize('pretrained', [True, False])
def test_resnext(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
