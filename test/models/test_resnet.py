import numpy as np
import pytest
from test.utils import convert_and_test
from torchvision.models import resnet18
from torch import nn


@pytest.mark.parametrize('model_class', [resnet18])
@pytest.mark.parametrize('pretrained', [True])
def test_resnet18(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
