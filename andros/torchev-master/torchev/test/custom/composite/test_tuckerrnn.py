import pytest
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.custom.composite.tuckerrnn import StatefulTuckerGRUCell, StatefulTuckerLSTMCell 

ENC_LEN = [30, 10, 8, 7, 4]
HID_SIZE = 128
IN_MODES = [4, 4, 4]
OUT_MODES = [4, 8, 4]
RANKS = [2,2,2]
BATCH, MAX_ENC_LEN, ENC_DIM = 64, max(ENC_LEN), int(np.prod(IN_MODES))

def test_ttgru() :
    input = Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))
    rnncell = StatefulTuckerGRUCell(IN_MODES, OUT_MODES, RANKS)
    assert rnncell.state is None
    res = rnncell(input[:, 0, :].contiguous())
    assert rnncell.state is not None

def test_ttlstm() :
    input = Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))
    rnncell = StatefulTuckerLSTMCell(IN_MODES, OUT_MODES, RANKS)
    assert rnncell.state is None
    res = rnncell(input[:, 0, :].contiguous())
    assert rnncell.state is not None
