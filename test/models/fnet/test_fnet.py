import numpy as np
import pytest
from test.utils import convert_and_test, NP_SEED
from test.models.fnet.conv3dnet import Net


@pytest.mark.slow
@pytest.mark.parametrize('pretrained', [True])
def test_fnet(pretrained):
    np.random.seed(seed=NP_SEED)
    model = Net()
    model.eval()

    input_np = np.random.uniform(0, 1, (1, 1, 32, 64, 64))
    error = convert_and_test(model, input_np, verbose=False, should_transform_inputs=True, epsilon=2 * 10 ** (-5))
