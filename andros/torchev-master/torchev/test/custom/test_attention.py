import pytest
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torchev.custom import MLPAttention, MLPHistoryAttention, \
        BilinearAttention, GMMAttention, \
        LocalGMMScorerAttention, BernoulliMonotonicAttention, \
        MultiheadKVQAttention, DotProductAttention

BATCH, MAX_ENC_LEN, ENC_DIM = 5, 10, 4 
DEC_DIM = 3
ENC_LEN = [6, 4, 2, 10, 7]

def create_ctx(dim=ENC_DIM) :
    return Variable(torch.randn(BATCH, MAX_ENC_LEN, dim), requires_grad=True)

def create_query(dim=DEC_DIM) :
    return Variable(torch.randn(BATCH, dim))

def create_mask() :
    mask = torch.ByteTensor(BATCH, MAX_ENC_LEN).fill_(False)
    for ii in range(BATCH) :
        mask[ii, 0:ENC_LEN[ii]] = True
    mask = Variable(mask)
    return mask

class TestMLPAttention :
    def test_attention_mask(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = MLPAttention(ENC_DIM, DEC_DIM)
        res = l_att(input)
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        res_ctx = res['expected_ctx'].sum()
        res_ctx.backward()
        pass
    pass

class TestMLPHistoryAttention :
    def test_attention_mask(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = MLPHistoryAttention(ENC_DIM, DEC_DIM)
        res = l_att(input)
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        res = l_att(input)
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        res_ctx = res['expected_ctx'].sum()
        res_ctx.backward()
        pass
    pass

class TestMultiheadKVQAttention :
    def test_attention_mask(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = MultiheadKVQAttention(ENC_DIM, DEC_DIM)
        res = l_att(input)
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        res_ctx = res['expected_ctx'].sum()
        res_ctx.backward()
        pass
    pass

class TestGMMAttention :
    def test_attention_mask(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = GMMAttention(ENC_DIM, DEC_DIM, n_mixtures=2)
        res = l_att(input)
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        res_ctx = res['expected_ctx'].sum()
        res_ctx.backward()
        pass
    def test_attention_mask_twostep(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = GMMAttention(ENC_DIM, DEC_DIM, n_mixtures=2)
        res = l_att(input)
        res_state = l_att.state
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        input['att_info'] = res
        res_2 = l_att(input)
        assert np.all(l_att.state['stat']['kappa'].data.numpy() > res_state['stat']['kappa'].data.numpy())
        res_ctx = res_2['expected_ctx'].sum()
        res_ctx.backward()
        pass
    def test_attention_mask_twostep_prune_ind(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = GMMAttention(ENC_DIM, DEC_DIM, n_mixtures=2, prune_range=[-2, 2], prune_type='ind')
        res = l_att(input)
        res_state = l_att.state
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        # assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        input['att_info'] = res
        res_2 = l_att(input)
        assert np.all(l_att.state['stat']['kappa'].data.numpy() > res_state['stat']['kappa'].data.numpy())
        res_ctx = res_2['expected_ctx'].sum()
        res_ctx.backward()
        pass
    def test_attention_mask_twostep_prune_exp(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = GMMAttention(ENC_DIM, DEC_DIM, n_mixtures=2, prune_range=[-2, 2], prune_type='expected',  normalize_alpha=True)
        res = l_att(input)
        res_state = l_att.state
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        # assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        input['att_info'] = res
        res_2 = l_att(input)
        assert np.all(l_att.state['stat']['kappa'].data.numpy() > res_state['stat']['kappa'].data.numpy())
        res_ctx = res_2['expected_ctx'].sum()
        res_ctx.backward()
        pass
    pass

class TestLocalGMMScorerAttention :
    def test_attention_mask(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = LocalGMMScorerAttention(ENC_DIM, DEC_DIM)
        res = l_att(input)
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        res_ctx = res['expected_ctx'].sum()
        res_ctx.backward()
        pass
    def test_attention_mask_twostep(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = LocalGMMScorerAttention(ENC_DIM, DEC_DIM)
        res = l_att(input)
        res_state = l_att.state
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        input['att_info'] = res
        res_2 = l_att(input)
        assert np.all(l_att.state['stat']['kappa'].data.numpy() > res_state['stat']['kappa'].data.numpy())
        res_ctx = res_2['expected_ctx'].sum()
        res_ctx.backward()
        pass
    def test_attention_mask_twostep_prune(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = LocalGMMScorerAttention(ENC_DIM, DEC_DIM, prune_range=[-3, 3], beta_val=0.222)
        res = l_att(input)
        res_state = l_att.state
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        input['att_info'] = res
        res_2 = l_att(input)
        assert np.all(l_att.state['stat']['kappa'].data.numpy() > res_state['stat']['kappa'].data.numpy())
        res_ctx = res_2['expected_ctx'].sum()
        res_ctx.backward()
        pass
    pass

class TestBilinearAttention :
    def test_attention_mask(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = BilinearAttention(ENC_DIM, DEC_DIM)
        res = l_att(input)
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        res_ctx = res['expected_ctx'].sum()
        res_ctx.backward()
        pass
    pass

class TestDotProductAttention :
    def test_attention_mask(self) :
        input = {'ctx':create_ctx(ENC_DIM), 'query':create_query(ENC_DIM), 'mask':create_mask()}
        l_att = DotProductAttention(ENC_DIM, ENC_DIM)
        res = l_att(input)
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        res_ctx = res['expected_ctx'].sum()
        res_ctx.backward()
        pass
    pass

class TestGoogleMonotonicAttention :
    def test_attention_mask(self) :
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        l_att = BernoulliMonotonicAttention(ENC_DIM, DEC_DIM)
        res = l_att(input)
        input = {'ctx':create_ctx(), 'query':create_query(), 'mask':create_mask()}
        res = l_att(input)
        res_prob_nonzero = (res['p_ctx'].data == 0.0).sum(1, keepdim=True).numpy().flatten()
        assert np.allclose(MAX_ENC_LEN-res_prob_nonzero, np.array(ENC_LEN))
        res_ctx = res['expected_ctx'].sum()
        res_ctx.backward()
        pass
    pass
