import numpy as np
import pytest
from torchvision.models import googlenet

from test.utils import convert_and_test


@pytest.mark.slow
@pytest.mark.parametrize('pretrained', [True, False])
def test_googlenet(pretrained):
    model = googlenet(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
