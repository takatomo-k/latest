import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torch.nn import init
import numpy as np
class BaseAttention(Module) :
    NEG_NUM = -10000.0
    def __init__(self) :
        super(BaseAttention, self).__init__()
        self.state = None

    def apply_mask(self, score, mask, mask_value=NEG_NUM) :
        # TODO inplace masked_fill_
        return score.masked_fill(mask == 1, mask_value)

    def calc_expected_context(self, p_ctx, ctx) :
        """
        p_ctx = (batch x srcL)
        ctx = (batch x srcL x dim)
        """
        p_ctx_3d = p_ctx.unsqueeze(1) # (batch x 1 x enc_len)
        expected_ctx = torch.bmm(p_ctx_3d, ctx).squeeze(1) # (batch x dim)
        return expected_ctx

    def reset(self) :
        self.state = None
        pass

class DotProductAttention(BaseAttention) :
    def __init__(self, normalize=True) :
        super().__init__()
        self.normalize = normalize

    def __call__(self, ctx, query,mask=None) :
        #import pdb; pdb.set_trace()
        batch, enc_len, enc_dim = ctx.size()
        score_ctx = torch.bmm(ctx, query.unsqueeze(-1)).squeeze(-1)
        if self.normalize :
            score_ctx = F.normalize(score_ctx,p=2,dim=-1)#score_ctx / self.denominator
        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = F.softmax(score_ctx, dim=-1)
        expected_ctx = self.calc_expected_context(p_ctx, ctx)
        return expected_ctx, p_ctx

class MLPAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, att_hid_size=256, act_fn=F.tanh,
            normalize=False) :
        super(MLPAttention, self).__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.att_hid_size = att_hid_size
        self.act_fn = act_fn
        self.lin_in2proj = nn.Linear(ctx_size + query_size, att_hid_size)
        if normalize :
            self.lin_proj2score = nn.Linear(att_hid_size, 1)
        else :
            self.lin_proj2score = nn.utils.weight_norm(nn.Linear(att_hid_size, 1))
        self.out_features = self.ctx_size
        self.normalize = normalize
        pass
    def __call__(self,ctx,query,mask=None) :
        batch, enc_len, enc_dim = ctx.size()
        #import pdb; pdb.set_trace()
        query_size =query.size(-1)
        combined_input = torch.cat([ctx, query.unsqueeze(1).expand(batch, enc_len, query_size)], -1) # batch x enc_len x (enc_dim + dec_dim) #
        combined_input_2d = combined_input.view(batch * enc_len, -1)
        score_ctx = self.lin_proj2score(F.tanh(self.lin_in2proj(combined_input_2d)))
        score_ctx = score_ctx.view(batch, enc_len) # batch x enc_len #
        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = F.softmax(score_ctx, dim=-1)
        expected_ctx = self.calc_expected_context(p_ctx, ctx)
        return expected_ctx, p_ctx
