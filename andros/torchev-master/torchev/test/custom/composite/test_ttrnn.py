import pytest
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.custom.composite.ttrnn import StatefulTTGRUCell, StatefulTTLSTMCell

ENC_LEN = [30, 10, 8, 7, 4]
HID_SIZE = 128
IN_MODES = [4, 4, 4]
OUT_MODES = [4, 8, 4]
RANKS = [1, 3, 3, 1]
BATCH, MAX_ENC_LEN, ENC_DIM = 64, max(ENC_LEN), int(np.prod(IN_MODES))

def test_ttgru() :
    input = Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))
    rnncell = StatefulTTGRUCell(IN_MODES, OUT_MODES, RANKS)
    res = rnncell(input[:, 0, :].contiguous())
    pass

def test_ttlstm() :
    input = Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))
    rnncell = StatefulTTLSTMCell(IN_MODES, OUT_MODES, RANKS)
    res = rnncell(input[:, 0, :].contiguous())
    pass
