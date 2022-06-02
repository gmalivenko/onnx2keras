import numpy as np
import pytest
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2
from torch import nn
from test.utils import convert_and_test


@pytest.mark.parametrize('model_class', [resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2,
                                         wide_resnet101_2])
@pytest.mark.parametrize('pretrained', [True, False])
def test_resnet18(pretrained, model_class):
    model = model_class(pretrained=pretrained)
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
