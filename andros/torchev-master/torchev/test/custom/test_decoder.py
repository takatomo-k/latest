import pytest
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.custom import MLPAttention, BilinearAttention
from torchev.custom import StandardDecoder

BATCH, MAX_ENC_LEN, ENC_DIM = 5, 10, 2
DEC_DIM = 3
ENC_LEN = [6, 4, 2, 10, 7]
DEC_IN_SIZE = 3

def create_ctx() :
    return Variable(torch.randn(BATCH, MAX_ENC_LEN, ENC_DIM))

def create_query() :
    return Variable(torch.randn(BATCH, DEC_DIM))

def create_mask() :
    mask = torch.ByteTensor(BATCH, MAX_ENC_LEN).fill_(False)
    for ii in range(BATCH) :
        mask[ii, 0:ENC_LEN[ii]] = True
    mask = Variable(mask)
    return mask

def create_dec_in() :
    return Variable(torch.randn(BATCH, DEC_IN_SIZE))

def create_dec_mask() :
    return Variable(torch.FloatTensor(BATCH).bernoulli_().float())

class TestStandardDecoder :
    def test_standard_decoder_with_att_mlp(self) :
        self._test_standard_decoder_with_att({'type':'mlp'})
        pass

    def test_standard_decoder_with_att_bilinear(self) :
        self._test_standard_decoder_with_att({'type':'bilinear'})
        pass

    def _test_standard_decoder_with_att(self, cfg) :
        l_dec = StandardDecoder(cfg, ENC_DIM, DEC_IN_SIZE, rnn_sizes=[DEC_DIM, DEC_DIM], ctx_proj_size=8)
        ctx = create_ctx()
        ctx_mask = create_mask()
        l_dec.set_ctx(ctx, ctx_mask)
        assert l_dec.ctx_proj_prev is None
        res = l_dec(create_dec_in(), create_dec_mask())
        assert l_dec.ctx_proj_prev is not None
        res = l_dec(create_dec_in(), create_dec_mask())
        assert l_dec.ctx_proj_prev is not None
        res_ctx = res['dec_output'].sum()
        res_ctx.backward()
        pass
    
    pass
