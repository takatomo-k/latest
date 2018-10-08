import pytest
import numpy as np
import torch

from torch.autograd import Variable
from torchev.nn.modules.candecomp import CPLinear

BATCH = 10
IN_MODES = [4, 8, 4, 8]
OUT_MODES = [2, 4, 2, 4]
ORDER = 2

class TestCPLinear(object) :
    def test_cplinear(self) :
        input = Variable(torch.randn(BATCH, int(np.prod(IN_MODES))))
        cplin_lyr = CPLinear(IN_MODES, OUT_MODES, ORDER, bias=True)
        out = cplin_lyr(input)
        assert out.size() == (BATCH, int(np.prod(OUT_MODES)))
        pass
    pass
