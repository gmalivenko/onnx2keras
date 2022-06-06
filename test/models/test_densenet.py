import numpy as np
import pytest
from torch import nn
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('model_class', [densenet121, densenet161, densenet169, densenet201])
@pytest.mark.parametrize('pretrained', [True, False])
def test_densenet(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
