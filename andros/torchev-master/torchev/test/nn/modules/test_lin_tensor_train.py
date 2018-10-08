import pytest
import numpy as np
import torch

from torch.autograd import Variable
from torchev.nn.modules.tensor_train import TTLinear

BATCH = 10
IN_MODES = [4, 8, 4, 8]
OUT_MODES = [2, 4, 2, 4]
RANKS = [1, 3, 3, 3, 1]

class TestTTLinear(object) :
    def test_ttlinear(self) :
        input = Variable(torch.randn(BATCH, int(np.prod(IN_MODES))))
        ttlin_lyr = TTLinear(IN_MODES, OUT_MODES, RANKS, bias=True)
        out = ttlin_lyr(input)
        assert out.size() == (BATCH, int(np.prod(OUT_MODES)))
        pass
    pass
