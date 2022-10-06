import numpy as np
import pytest
from torch import nn
from test.utils import convert_and_test, NP_SEED
from torchvision.models import efficientnet_b0




@pytest.mark.slow
@pytest.mark.parametrize('model_class', [efficientnet_b0])
@pytest.mark.parametrize('pretrained', [True])
def test_efficientnet(pretrained, model_class):
    np.random.seed(seed=NP_SEED)
    model = model_class(pretrained=pretrained)
    model = nn.Sequential(
        model,
        nn.Softmax(dim=1)
    )
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
