import numpy as np
import pytest
from torchvision.models import resnet18

from test.utils import convert_and_test


@pytest.mark.parametrize('change_ordering', [False])
def test_resnet18(change_ordering):
    model = resnet18()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
