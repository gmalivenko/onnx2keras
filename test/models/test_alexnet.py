import numpy as np
from torchvision.models import alexnet

from test.utils import convert_and_test


def test_alexnet():
    model = alexnet()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True)
