import pytest
import numpy as np
import torch

from torch.autograd import Variable
from torchev.nn.modules.tucker import TuckerLinear

BATCH = 10
IN_MODES = [4, 8, 4, 8]
OUT_MODES = [2, 4, 2, 4]
RANKS = [2, 2, 2, 2]

class TestTuckerLinear(object) :
    def test_cplinear(self) :
        input = Variable(torch.randn(BATCH, int(np.prod(IN_MODES))))
        cplin_lyr = TuckerLinear(IN_MODES, OUT_MODES, RANKS, bias=True, cache=True)
        out = cplin_lyr(input)
        assert out.size() == (BATCH, int(np.prod(OUT_MODES)))
        pass
    pass
