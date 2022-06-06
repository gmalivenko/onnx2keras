import numpy as np
import pytest
from torchvision.models import convnext_base, convnext_tiny, convnext_large, convnext_small
from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [convnext_base, convnext_tiny, convnext_large, convnext_small])
@pytest.mark.parametrize('pretrained', [True, False])
def test_convnext(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
