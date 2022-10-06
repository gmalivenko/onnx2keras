import numpy as np
import pytest
from torch import nn
from test.utils import convert_and_test
from torchvision.models import mobilenet_v3_small



@pytest.mark.slow
@pytest.mark.parametrize('model_class', [mobilenet_v3_small])
@pytest.mark.parametrize('pretrained', [True])
def test_mobilenetv3(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 124, 124))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
