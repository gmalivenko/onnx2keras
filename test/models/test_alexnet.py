import numpy as np
import pytest
from test.utils import convert_and_test
from torchvision.models import alexnet
from test.utils import NP_SEED

@pytest.mark.parametrize('pretrained', [True])
def test_alexnet(pretrained):
    np.random.seed(seed=NP_SEED)
    model = alexnet(pretrained=pretrained)
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
