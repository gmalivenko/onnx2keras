import numpy as np
import pytest
from torchvision.models.densenet import densenet121

from test.utils import convert_and_test


@pytest.mark.slow
def test_densenet():
    model = densenet121()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
