import numpy as np
import pytest
from torchvision.models import squeezenet1_0, squeezenet1_1

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [squeezenet1_1, squeezenet1_0])
@pytest.mark.parametrize('pretrained', [True, False])
def test_squeezenet(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
