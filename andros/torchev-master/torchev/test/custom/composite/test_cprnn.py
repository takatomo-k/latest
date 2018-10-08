import pytest
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.custom.composite.cprnn import StatefulCPGRUCell, StatefulCPLSTMCell 

ENC_LEN = [30, 10, 8, 7, 4]
HID_SIZE = 128
IN_MODES = [4, 4, 4]
OUT_MODES = [4, 8, 4]
ORDER = 2
BATCH, MAX_ENC_LEN, ENC_DIM = 64, max(ENC_LEN), int(np.prod(IN_MODES))

def test_ttgru() :
    input = Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))
    rnncell = StatefulCPGRUCell(IN_MODES, OUT_MODES, ORDER)
    assert rnncell.state is None
    res = rnncell(input[:, 0, :].contiguous())
    assert rnncell.state is not None

def test_ttlstm() :
    input = Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))
    rnncell = StatefulCPLSTMCell(IN_MODES, OUT_MODES, ORDER)
    assert rnncell.state is None
    res = rnncell(input[:, 0, :].contiguous())
    assert rnncell.state is not None
