import numpy as np
import pytest
from test.utils import convert_and_test
from torchvision.models import googlenet



@pytest.mark.slow
@pytest.mark.parametrize('pretrained', [True])
@pytest.mark.skip(reason="Fails on CI init")
def test_googlenet(pretrained):
    model = googlenet(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
