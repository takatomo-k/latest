import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torch.nn import init
import numpy as np
from ..utils.helper import torchauto, tensorauto
from ..generator import generator_act_fn, generator_attention
from ..custom.composite import MultiscaleConv1d
from ..nn.modules.embedding import PositionEmbedding

class BaseAttention(Module) :
    NEG_NUM = -10000.0
    def __init__(self) :
        super(BaseAttention, self).__init__()
        self.state = None

    def apply_mask(self, score, mask, mask_value=NEG_NUM) :
        # TODO inplace masked_fill_
        return score.masked_fill(mask == 0, mask_value)

    def forward_single(self, input) :
        """
        key : batch x src_len x dim_k
        query : batch x dim_q
        """
        raise NotImplementedError()

    def forward_multiple(self, input) :
        """
        key : batch x src_len x dim_k
        query : batch x tgt_len x dim_q
        """
        raise NotImplementedError()

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
    def __init__(self, ctx_size, query_size, normalize=False) :
        """
        Args:
            normalize: (bool) if True, the variance from output score normalized into 1
        """
        super().__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        assert self.ctx_size == self.query_size, "ctx_size must be same as query_size"
        self.normalize = normalize
        if self.normalize :
            self.denominator = np.sqrt(query_size)
        assert self.ctx_size == self.query_size

    def forward(self, input) :
        if input['query'].dim() == 2 :
            return self.forward_single(input)
        else :
            return self.forward_multiple(input)

    def forward_single(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        # calc_val (skip calculate expected context if False)
        calc_val = input.get('calc_val', True) # for scoring only
        batch, enc_len, enc_dim = ctx.size()
        score_ctx = torch.bmm(ctx, query.unsqueeze(-1)).squeeze(-1)
        if self.normalize :
            score_ctx = score_ctx / self.denominator
        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = F.softmax(score_ctx, dim=-1)
        if calc_val :
            expected_ctx = self.calc_expected_context(p_ctx, ctx)
        else :
            expected_ctx = None
        return expected_ctx, p_ctx

    def forward_multiple(self, input) :
        query = input['query'] # batch x dec_len x dec_dim #
        assert query.dim() == 3
        result = []
        for ii in range(query.shape[1]) :
            _input_ii = dict(input)
            _input_ii['query'] = query[:, ii]
            result.append(self.forward_single(_input_ii))
        result = list(zip(*result))
        return (torch.stack(item, dim=1) for item in result)


    def __call__(self, *input, **kwargs) :
        result = super().__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}

class BilinearAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, normalize=False) :
        """
        Args:
            normalize: (bool) if True, the variance from output score normalized into 1
        """
        super(BilinearAttention, self).__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.W = nn.Linear(query_size, ctx_size, bias=False)
        self.out_features = self.ctx_size
        self.normalize = normalize
        if self.normalize :
            self.denominator = np.sqrt(ctx_size * query_size)
        pass

    def forward(self, input) :
        if input['query'].dim() == 2 :
            return self.forward_single(input)
        else :
            return self.forward_multiple(input)

    def forward_single(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()
        score_ctx = ctx.bmm(self.W(query).unsqueeze(2))
        score_ctx = score_ctx.squeeze(2)
        if self.normalize :
            score_ctx = score_ctx / self.denominator
        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = F.softmax(score_ctx, dim=-1)
        expected_ctx = self.calc_expected_context(p_ctx, ctx)
        return expected_ctx, p_ctx

    def forward_multiple(self, input) :
        query = input['query'] # batch x dec_len x dec_dim #
        assert query.dim() == 3
        result = []
        for ii in range(query.shape[1]) :
            _input_ii = dict(input)
            _input_ii['query'] = query[:, ii]
            result.append(self.forward_single(_input_ii))
        result = list(zip(*result))
        return (torch.stack(item, dim=1) for item in result)

    def __call__(self, *input, **kwargs) :
        result = super(BilinearAttention, self).__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}

class MLPAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, att_hid_size=256) :
        super(MLPAttention, self).__init__()
        self.lin_in2proj = nn.Linear(ctx_size + query_size, att_hid_size)
        self.lin_proj2score = nn.Linear(att_hid_size, 1)

    def forward(self, input) :
        if input['query'].dim() == 2 :
            return self.forward_single(input)
        else :
            return self.forward_multiple(input)

    def forward_single(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()

        combined_input = torch.cat([ctx, query.unsqueeze(1).expand(batch, enc_len, self.query_size)], 2) # batch x enc_len x (enc_dim + dec_dim) #
        combined_input_2d = combined_input.view(batch * enc_len, -1)
        score_ctx = self.lin_proj2score(self.act_fn(self.lin_in2proj(combined_input_2d)))
        score_ctx = score_ctx.view(batch, enc_len) # batch x enc_len #
        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = F.softmax(score_ctx, dim=-1)
        expected_ctx = self.calc_expected_context(p_ctx, ctx)
        return expected_ctx, p_ctx

    def forward_multiple(self, input) :
        query = input['query'] # batch x dec_len x dec_dim #
        assert query.dim() == 3
        result = []
        for ii in range(query.shape[1]) :
            _input_ii = dict(input)
            _input_ii['query'] = query[:, ii]
            result.append(self.forward_single(_input_ii))
        result = list(zip(*result))
        return (torch.stack(item, dim=1) for item in result)

    def __call__(self, *input, **kwargs) :
        result = super(MLPAttention, self).__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}
    pass

class MLPHistoryAttention(BaseAttention) :
    """
    Additional information from previous attention prob vector
    """
    def __init__(self, ctx_size, query_size, att_hid_size=256, act_fn=F.tanh,
            history_conv_ch=[64, 64, 64], history_conv_ksize=[32, 64, 128],
            normalize=False) :
        super().__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.att_hid_size = att_hid_size
        self.act_fn = act_fn
        self.history_conv_ch = history_conv_ch
        self.history_conv_ksize = history_conv_ksize
        self.lin_in2proj = nn.Linear(ctx_size + query_size, att_hid_size)
        if normalize :
            self.lin_proj2score = nn.Linear(att_hid_size, 1)
        else :
            self.lin_proj2score = nn.utils.weight_norm(nn.Linear(att_hid_size, 1))

        # aux history convolution #
        self.history_conv = MultiscaleConv1d(1, history_conv_ch, history_conv_ksize)
        self.history_conv2proj = nn.Linear(sum(history_conv_ch), att_hid_size)

        self.out_features = self.ctx_size
        pass

    def _init_p_ctx(self, batch, enc_len) :
        _p_ctx = torchauto(self).FloatTensor(batch, enc_len).zero_()
        _p_ctx[:, 0] = 1.0
        return Variable(_p_ctx)

    def forward(self, input) :
        if input['query'].dim() == 2 :
            return self.forward_single(input)
        else :
            return self.forward_multiple(input)

    def forward_single(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        # initialize prev attention if not available #
        batch, enc_len = ctx.shape[0:2]
        if self.state is None :
            self.state = {'p_ctx':self._init_p_ctx(batch, enc_len)}
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()
        combined_input = torch.cat([ctx, query.unsqueeze(1).expand(batch, enc_len, self.query_size)], 2) # batch x enc_len x (enc_dim + dec_dim) #
        # aux history convolution #
        _history_pre_score = self.history_conv(self.state['p_ctx'].unsqueeze(1)).transpose(1, 2) # batch x enc_len x num_ch #
        _history_pre_score = self.history_conv2proj(_history_pre_score) # batch x enc_len x hid_size #
        score_ctx = self.lin_proj2score(self.act_fn(self.lin_in2proj(combined_input) + _history_pre_score))
        score_ctx = score_ctx.view(batch, enc_len) # batch x enc_len #

        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = F.softmax(score_ctx, dim=-1)
        expected_ctx = self.calc_expected_context(p_ctx, ctx)

        # save p_ctx for next step calculation
        self.state['p_ctx'] = p_ctx
        return expected_ctx, p_ctx

    def forward_multiple(self, input) :
        query = input['query'] # batch x dec_len x dec_dim #
        assert query.dim() == 3
        result = []
        for ii in range(query.shape[1]) :
            _input_ii = dict(input)
            _input_ii['query'] = query[:, ii]
            result.append(self.forward_single(_input_ii))
        result = list(zip(*result))
        return (torch.stack(item, dim=1) for item in result)

    def __call__(self, *input, **kwargs) :
        result = super().__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}
    pass

class MultiheadKVQAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, num_heads=8, proj_dim=512, proj_bias=True, proj_fn=F.tanh,
            sub_att_lyr_cfg={'type':'dot', 'kwargs':{'normalize':True}}) :
        super().__init__()
        assert proj_dim % num_heads == 0, "num_heads must be a factor from proj_dim"
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.num_heads = num_heads
        self.proj_dim = proj_dim

        self.dim_per_head = proj_dim // num_heads

        self.proj_key_lyr = nn.Linear(ctx_size, proj_dim, bias=proj_bias)
        self.proj_value_lyr = nn.Linear(ctx_size, proj_dim, bias=proj_bias)
        self.proj_query_lyr = nn.Linear(query_size, proj_dim, bias=proj_bias)
        self.final_proj_lyr = nn.Linear(proj_dim, proj_dim)
        self.proj_fn = generator_act_fn(proj_fn) if isinstance(proj_fn, (str)) else proj_fn

        _sub_att_lyr_cfg = dict(sub_att_lyr_cfg)
        _sub_att_lyr_cfg['kwargs'] = {'ctx_size':self.dim_per_head, 'query_size':self.dim_per_head}
        self.sub_att_lyr = generator_attention(_sub_att_lyr_cfg)
        self.out_features = proj_dim

    def forward(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        # TODO: implement self-attention
        query = input['query'] # batch x dec_dim OR batch x dec_len x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()

        proj_key = self.proj_key_lyr(ctx) # batch x enc_len x proj_dim #
        proj_value = self.proj_value_lyr(ctx) # batch x enc_len x proj_dim #
        proj_query = self.proj_query_lyr(query) # batch x proj_dim #

        # reshape
        proj_key = proj_key.view(batch, enc_len, self.num_heads, self.dim_per_head).transpose(1, 2).contiguous().view(batch * self.num_heads, enc_len, self.dim_per_head)
        proj_value = proj_value.view(batch, enc_len, self.num_heads, self.dim_per_head).transpose(1, 2).contiguous().view(batch * self.num_heads, enc_len, self.dim_per_head)
        proj_query = proj_query.view(batch*self.num_heads, self.dim_per_head)
        mask = mask.unsqueeze(-1).expand(batch, enc_len, self.num_heads).transpose(1, 2).contiguous().view(batch * self.num_heads, enc_len)

        sub_att_input = {'ctx':proj_key, 'query':proj_query, 'mask':mask, 'calc_val':False}
        sub_att_res = self.sub_att_lyr(sub_att_input) #  TODO: state save
        sub_att_expected_ctx = self.calc_expected_context(sub_att_res['p_ctx'], proj_value).view(batch, self.num_heads * self.dim_per_head)
        sub_att_p_ctx = sub_att_res['p_ctx'].view(batch, self.num_heads, enc_len).mean(dim=1)
        assert sub_att_p_ctx.shape == (batch, enc_len) # TODO remove

        final_proj_ctx = self.proj_fn(self.final_proj_lyr(sub_att_expected_ctx))
        return final_proj_ctx, sub_att_p_ctx

    def __call__(self, *input, **kwargs) :
        result = super().__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}
    pass

class MultiheadKVQPosAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, num_heads=8,
            proj_dim=512, proj_bias=True, proj_fn=F.tanh,
            sub_att_lyr_cfg={'type':'dot', 'kwargs':{'normalize':True}},
            pos_emb_dim=128, pos_emb_type='rel') :
        super().__init__()
        assert proj_dim % num_heads == 0, "num_heads must be a factor from proj_dim"
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.num_heads = num_heads
        self.proj_dim = proj_dim

        self.dim_per_head = proj_dim // num_heads

        self.proj_key_lyr = nn.Linear(ctx_size, proj_dim, bias=proj_bias)
        self.proj_value_lyr = nn.Linear(ctx_size, proj_dim, bias=proj_bias)
        self.proj_query_lyr = nn.Linear(query_size, proj_dim, bias=proj_bias)
        self.final_proj_lyr = nn.Linear(proj_dim, proj_dim)
        self.proj_fn = generator_act_fn(proj_fn) if isinstance(proj_fn, (str)) else proj_fn

        self.pos_emb_lyr = PositionEmbedding(pos_emb_dim, max_len=4000, pos_emb_type='rel')

        _sub_att_lyr_cfg = dict(sub_att_lyr_cfg)
        _sub_att_lyr_cfg['kwargs'] = {'ctx_size':self.dim_per_head, 'query_size':self.dim_per_head}
        self.sub_att_lyr = generator_attention(_sub_att_lyr_cfg)
        self.out_features = proj_dim

    def forward(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        # TODO: implement self-attention
        query = input['query'] # batch x dec_dim OR batch x dec_len x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()

        proj_key = self.proj_key_lyr(ctx) # batch x enc_len x proj_dim #
        proj_value = self.proj_value_lyr(ctx) # batch x enc_len x proj_dim #
        proj_query = self.proj_query_lyr(query) # batch x proj_dim #

        # reshape
        proj_key = proj_key.view(batch, enc_len, self.num_heads, self.dim_per_head).transpose(1, 2).contiguous().view(batch * self.num_heads, enc_len, self.dim_per_head)
        proj_value = proj_value.view(batch, enc_len, self.num_heads, self.dim_per_head).transpose(1, 2).contiguous().view(batch * self.num_heads, enc_len, self.dim_per_head)
        proj_query = proj_query.view(batch*self.num_heads, self.dim_per_head)
        mask = mask.unsqueeze(-1).expand(batch, enc_len, self.num_heads).transpose(1, 2).contiguous().view(batch * self.num_heads, enc_len)

        sub_att_input = {'ctx':proj_key, 'query':proj_query, 'mask':mask, 'calc_val':False}
        sub_att_res = self.sub_att_lyr(sub_att_input) #  TODO: state save
        sub_att_expected_ctx = self.calc_expected_context(sub_att_res['p_ctx'], proj_value).view(batch, self.num_heads * self.dim_per_head)
        sub_att_p_ctx = sub_att_res['p_ctx'].view(batch, self.num_heads, enc_len).sum(dim=1)
        assert sub_att_p_ctx.shape == (batch, enc_len) # TODO remove

        final_proj_ctx = self.proj_fn(self.final_proj_lyr(sub_att_expected_ctx))
        return final_proj_ctx, sub_att_p_ctx

    def __call__(self, *input, **kwargs) :
        result = super().__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}
    pass

###########################
### MONOTONIC ATTENTION ###
###########################

# constant
ALPHA="alpha"
BETA="beta"
KAPPA="kappa"

class GMMAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, n_mixtures=5, monotonic=True,
            att_hid_size=256, act_fn=F.tanh,
            alpha_fn=torch.exp, beta_fn=torch.exp, kappa_fn=torch.exp,
            bprop_prev=False, alpha_bias=0, beta_bias=0, kappa_bias=-1,
            normalize_alpha=False, prune_range=None, prune_type=None) :
        """
        GMM attention (Alex Graves - synthesis network)
        alpha : mixture weight
        beta : width of window
        kappa : centre of window
        monotonic : option for strictly moving from left to right
        stat_bias : initialize alpha, beta, kappa bias
        Args :
            att_hid_size : int (use None to avoid projection layer)

        """
        super(GMMAttention, self).__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.n_mixtures = n_mixtures
        self.monotonic = monotonic

        self.att_hid_size = att_hid_size
        self.act_fn = act_fn
        self.alpha_fn=alpha_fn
        self.beta_fn=beta_fn
        self.kappa_fn=kappa_fn
        self.bprop_prev = bprop_prev
        self.alpha_bias = alpha_bias
        self.beta_bias = beta_bias
        self.kappa_bias = kappa_bias
        self.normalize_alpha = normalize_alpha
        self.prune_range = prune_range
        self.prune_type = prune_type
        self.out_features = self.ctx_size

        if self.att_hid_size is not None :
            self.lin_query2proj = nn.Linear(query_size, self.att_hid_size)
            self.lin_proj2stat = nn.Linear(self.att_hid_size, self.n_mixtures*3)
        else :
            self.lin_query2proj = lambda x : x
            self.act_fn = lambda x : x
            self.lin_proj2stat = nn.Linear(query_size, self.n_mixtures*3)
        init.constant(self.lin_proj2stat.bias[0:self.n_mixtures], alpha_bias)
        init.constant(self.lin_proj2stat.bias[self.n_mixtures:2*self.n_mixtures], beta_bias)
        init.constant(self.lin_proj2stat.bias[2*self.n_mixtures:3*self.n_mixtures], kappa_bias)
        pass

    def _init_stat(self, batch) :
        _zero_stat = torchauto(self).FloatTensor(batch, self.n_mixtures).zero_()
        return {ALPHA:Variable(_zero_stat.clone()),
                BETA:Variable(_zero_stat.clone()),
                KAPPA:Variable(_zero_stat.clone())}

    def forward(self, input) :

        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()

        if self.state is None :
            self.state = {'stat':self._init_stat(batch)}
        stat_prev = self.state['stat']

        if not self.bprop_prev :
            for stat_name in [ALPHA, BETA, KAPPA] :
                stat_prev[stat_name] = stat_prev[stat_name].detach()
        _alpha_prev, _beta_prev, _kappa_prev = stat_prev[ALPHA], stat_prev[BETA], stat_prev[KAPPA] # batch x n_mix #

        _alpha_beta_kappa = self.lin_proj2stat(self.act_fn(self.lin_query2proj(query)))
        _alpha, _beta, _kappa = _alpha_beta_kappa.split(self.n_mixtures, dim=1) # batch x n_mix #
        _alpha = self.alpha_fn(_alpha)
        _beta = self.beta_fn(_beta)
        _kappa = self.kappa_fn(_kappa)

        if self.normalize_alpha :
            _alpha = _alpha / _alpha.sum(1, keepdim=True)

        if self.monotonic :
            _kappa = _kappa + _kappa_prev

        pos_range = Variable(tensorauto(self, torch.arange(0, enc_len)). # enc_len #
                unsqueeze(0).expand(batch, enc_len). # batch x enc_len #
                unsqueeze(1).expand(batch, self.n_mixtures, enc_len)) # batch x n_mix x enc_len #

        alpha = _alpha.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)
        beta = _beta.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)
        kappa = _kappa.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)

        # generate pruning mask based on current kappa #
        if self.prune_type is not None :
            if self.prune_type == 'ind' :
                """
                ind : independent, for each Gaussian, user have independent mask pruning
                """
                prune_mask = torchauto(self).FloatTensor(batch, self.n_mixtures, enc_len).zero_()

                for ii in range(batch) :
                    for jj in range(self.n_mixtures) :
                        _center = _kappa.data[ii, jj]
                        _left = round(_center) + self.prune_range[0]
                        _right = round(_center) + self.prune_range[1] + 1
                        if _right <= 0 or _left >= enc_len :
                            continue
                        else :
                            _left = max(0, _left)
                            _right = min(enc_len, _right)
                            prune_mask[ii, jj, _left:_right] = 1
                prune_mask = prune_mask.expand(batch, self.n_mixtures, enc_len)
            elif self.prune_type == 'expected' :
                """
                expected : calculate expected position from GMM and set mask pruning around it
                """
                assert self.normalize_alpha, "need normalize alpha to generate correct expected value"
                prune_mask = torchauto(self).FloatTensor(batch, enc_len).zero_()
                expected_val = torch.sum(_kappa * _alpha, dim=1, keepdim=True).squeeze(1) # batch #
                for ii in range(batch) :
                    _center = expected_val.data[ii]
                    _left = round(_center) + self.prune_range[0]
                    _right = round(_center) + self.prune_range[1] + 1
                    if _right <= 0 or _left >= enc_len :
                        continue
                    else :
                        _left = max(0, _left)
                        _right = min(enc_len, _right)
                        prune_mask[ii, _left:_right] = 1
                prune_mask = prune_mask.unsqueeze(1).expand(batch, self.n_mixtures, enc_len)
            else :
                raise ValueError()

            if mask is not None :
                prune_mask = prune_mask * mask.data.unsqueeze(1).expand(batch, self.n_mixtures, enc_len).float()
            prune_mask = Variable(prune_mask)
        else :
            prune_mask = None
        score_pos = alpha * torch.exp(-beta * (pos_range - kappa)**2)
        if self.prune_type is not None :
            score_pos = score_pos * prune_mask
            pass
        score_pos = torch.sum(score_pos, dim=1).squeeze(1)
        if mask is not None :
            score_pos = self.apply_mask(score_pos, mask, 0)

        # TODO normalize or not ? - original no normalize #
        p_ctx = score_pos
        expected_ctx = self.calc_expected_context(p_ctx, ctx)

        # save state
        self.state = {'stat':{ALPHA:_alpha, BETA:_beta, KAPPA:_kappa}}
        return expected_ctx, p_ctx

    def __call__(self, *input, **kwargs) :
        result = super(GMMAttention, self).__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return  {
                    "p_ctx":p_ctx,
                    "expected_ctx":expected_ctx
                }

    pass

class LocalGMMScorerAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, n_mixtures=1, monotonic=True,
            att_hid_size=256, act_fn=F.tanh,
            alpha_fn=torch.exp, beta_fn=torch.exp, kappa_fn=torch.exp,
            bprop_prev=False, alpha_bias=0, beta_bias=0, kappa_bias=-1,
            normalize_alpha=False,
            scorer_cfg={'type':'mlp'},
            normalize_scorer=False,
            prune_range=[-3, 3],
            normalize_post=True, # if True, sum(posterior) == 1 #
            beta_val=None, # if provided, beta_val will be fixed instead of input dependent #
            # if prune = 3 = (2*std_dev) then beta_val = 0.222 = (1.0 / (2*(1.5**2))) formula :  beta_val = 1/(2*sqrt(std_dev^2)) #
            kappa_val=None, # 'auto' mode : determined by model, N (int) : increment position every decode step #
            ignore_likelihood=False,
            ) :
        """
        GMM attention (Alex Graves - synthesis network)
        alpha : mixture weight
        beta : inverse width of window
        kappa : centre of window
        monotonic : option for strictly moving from left to right
        stat_bias : initialize alpha, beta, kappa bias
        Args :
            att_hid_size : int (use None to avoid projection layer)

        """
        super(LocalGMMScorerAttention, self).__init__()
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.n_mixtures = n_mixtures
        self.monotonic = monotonic

        self.att_hid_size = att_hid_size
        self.act_fn = act_fn

        self.alpha_fn = alpha_fn if not isinstance(alpha_fn, (str)) else generator_act_fn(alpha_fn)
        self.beta_fn = beta_fn if not isinstance(beta_fn, (str)) else generator_act_fn(beta_fn)
        self.kappa_fn = kappa_fn if not isinstance(kappa_fn, (str)) else generator_act_fn(kappa_fn)

        self.bprop_prev = bprop_prev
        self.alpha_bias = alpha_bias
        self.beta_bias = beta_bias
        self.kappa_bias = kappa_bias
        self.normalize_alpha = normalize_alpha


        self.scorer_cfg = scorer_cfg
        self.normalize_scorer = normalize_scorer
        if isinstance(prune_range, int) :
            self.prune_range = [-prune_range, prune_range]
        else :
            self.prune_range = prune_range
        assert self.prune_range is None or len(self.prune_range) == 2
        if self.prune_range is not None :
            assert self.prune_range[0] < 0 and self.prune_range[1] > 0

        self.normalize_post = normalize_post
        self.beta_val = beta_val
        if beta_val is not None :
            assert beta_val > 0
        self.ignore_likelihood = ignore_likelihood
        self.kappa_val = kappa_val

        if self.att_hid_size is not None :
            self.lin_query2proj = nn.Linear(query_size, self.att_hid_size)
            self.lin_proj2stat = nn.Linear(self.att_hid_size, self.n_mixtures*3)
        else :
            self.lin_query2proj = lambda x : x
            self.act_fn = lambda x : x
            self.lin_proj2stat = nn.Linear(query_size, self.n_mixtures*3)
        init.constant(self.lin_proj2stat.bias[0:self.n_mixtures], alpha_bias)
        init.constant(self.lin_proj2stat.bias[self.n_mixtures:2*self.n_mixtures], beta_bias)
        init.constant(self.lin_proj2stat.bias[2*self.n_mixtures:3*self.n_mixtures], kappa_bias)

        # scorer #
        if scorer_cfg['type'] == 'mlp' :
            self.scorer = MLPAttention(ctx_size, query_size, att_hid_size, act_fn)
            self.scorer_module = nn.ModuleList(
                    [nn.Linear(ctx_size+query_size, att_hid_size),
                    nn.Linear(att_hid_size, 1)])
            def _fn_scorer(_ctx, _query) :
                batch, enc_len, enc_dim = _ctx.size()
                combined_input = torch.cat([_ctx, _query.unsqueeze(1).expand(batch, enc_len, query_size)], 2)
                combined_input_2d = combined_input.view(batch * enc_len, -1)
                score_ctx = self.scorer_module[1](self.act_fn(self.scorer_module[0](combined_input_2d)))
                score_ctx = score_ctx.view(batch, enc_len)
                return score_ctx
            self.scorer_fn = _fn_scorer

        else :
            raise NotImplementedError()

        self.out_features = self.ctx_size
        pass

    def _init_stat(self, batch) :
        _zero_stat = torchauto(self).FloatTensor(batch, self.n_mixtures).zero_()
        return {ALPHA:Variable(_zero_stat.clone()),
                BETA:Variable(_zero_stat.clone()),
                KAPPA:Variable(_zero_stat.clone())}

    def forward(self, input) :

        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()

        if self.state is None :
            self.state = {'stat':self._init_stat(batch)}
        stat_prev = self.state['stat'] # dict of 3 batch x n_mix #

        if not self.bprop_prev :
            for stat_name in [ALPHA, BETA, KAPPA] :
                stat_prev[stat_name] = stat_prev[stat_name].detach()
        _alpha_prev, _beta_prev, _kappa_prev = stat_prev[ALPHA], stat_prev[BETA], stat_prev[KAPPA] # batch x n_mix #

        _alpha_beta_kappa = self.lin_proj2stat(self.act_fn(self.lin_query2proj(query)))
        _alpha, _beta, _kappa = _alpha_beta_kappa.split(self.n_mixtures, dim=1) # batch x n_mix #
        _alpha = self.alpha_fn(_alpha)

        if self.beta_val is None :
            _beta = self.beta_fn(_beta)
        else : # replace with fixed beta value #
            _beta = Variable(torchauto(self).FloatTensor([self.beta_val])).expand_as(_beta)

        if self.kappa_val is None :
            _kappa = self.kappa_fn(_kappa)
        else : # replace with fixed kappa value #
            _kappa = Variable(torchauto(self).FloatTensor([self.kappa_val]).expand_as(_kappa))

        if self.normalize_alpha :
            _alpha = _alpha / _alpha.sum(1).expand_as(_alpha)

        if self.monotonic :
            _kappa = _kappa + _kappa_prev

        pos_range = Variable(tensorauto(self, torch.arange(0, enc_len)). # enc_len #
                unsqueeze(0).expand(batch, enc_len). # batch x enc_len #
                unsqueeze(1).expand(batch, self.n_mixtures, enc_len)) # batch x n_mix x enc_len #

        alpha = _alpha.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)
        beta = _beta.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)
        kappa = _kappa.unsqueeze(2).expand(batch, self.n_mixtures, enc_len)
        # calculate prior prob #
        prior_score = alpha * torch.exp(-beta * (pos_range - kappa)**2)

        # generate pruning mask based on current kappa #
        if self.prune_range is not None :
            prune_mask = torchauto(self).FloatTensor(batch, self.n_mixtures, enc_len).zero_()

            for ii in range(batch) :
                for jj in range(self.n_mixtures) :
                    _center = _kappa.data[ii, jj]
                    _left = round(_center) + self.prune_range[0]
                    _right = round(_center) + self.prune_range[1] + 1
                    if _right <= 0 or _left >= enc_len :
                        continue
                    else :
                        _left = max(0, _left)
                        _right = min(enc_len, _right)
                        prune_mask[ii, jj, _left:_right] = 1
            prune_mask = prune_mask.expand(batch, self.n_mixtures, enc_len)
            if mask is not None :
                prune_mask = prune_mask * mask.data.unsqueeze(1).expand(batch, self.n_mixtures, enc_len).float()
            prune_mask = Variable(prune_mask)
        else :
            prune_mask = None
        # import ipdb; ipdb.set_trace()

        if prune_mask is not None :
            # mask prior attention with pruning mask #
            prior_score = prior_score * prune_mask

        prior_score = torch.sum(prior_score, dim=1).squeeze(1)

        if mask is not None :
            prior_score = self.apply_mask(prior_score, mask, 0)


        # calculate posterior prob #
        if self.ignore_likelihood :
            # use prior as posterior #
            posterior_score = prior_score
        else :
            # generate scorer #
            # calculate likelihood prob #
            # TODO : normalize or not ? give prune_mask or mask ? #
            assert self.n_mixtures == 1
            likelihood_score = self.scorer_fn(ctx, query)
            if prune_mask is not None :
                likelihood_score = self.apply_mask(likelihood_score, prune_mask[:, 0, :])
            if self.normalize_scorer :
                likelihood_score = F.softmax(likelihood_score, dim=-1)
            else :
                likelihood_score = torch.exp(likelihood_score)
            # combine likelihood x prior score #
            posterior_score = likelihood_score * prior_score

        if self.prune_range is not None :
            posterior_score = self.apply_mask(posterior_score, prune_mask[:, 0, :], 0)

        # normalize posterior #
        if self.normalize_post :
            EPS = 1e-7
            posterior_score = posterior_score / (posterior_score.sum(1, keepdim=True) + EPS)

        # TODO normalize or not ? - original no normalize #
        p_ctx = posterior_score
        expected_ctx = self.calc_expected_context(p_ctx, ctx)

        # save state
        self.state = {'stat':{ALPHA:_alpha, BETA:_beta, KAPPA:_kappa}}
        return expected_ctx, p_ctx

    def __call__(self, *input, **kwargs) :
        result = super(LocalGMMScorerAttention, self).__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return  {
                    "p_ctx":p_ctx,
                    "expected_ctx":expected_ctx
                }

    pass

### GOOGLE MONOTONIC ATTENTION ###
def _monotonic_probability_fn(score, previous_att, sigmoid_noise, mode) :
    """
    Ref : Online and Linear-Time Attention by Enforcing Monotonic Alignments

    Args:
        score : unnormalized attention score [batch, seq_len]
    """
    if sigmoid_noise > 0 :
        noise = Variable(tensorauto(score, torch.randn(score.size())))
        score += sigmoid_noise * noise

    if mode == 'hard' :
        raise NotImplementedError()
    else :
        p_choose_i = F.sigmoid(score)

    return _monotonic_attention(p_choose_i, previous_att, mode)
    pass

def _monotonic_attention(p_choose_i, previous_att, mode) :
    batch, seq_len = p_choose_i.size()[0:2]
    if mode == 'recursive' :
        shifted_1mp_choose_i = torch.cat(
                [Variable(torchauto(p_choose_i).FloatTensor(batch, 1).fill_(1)), 1-p_choose_i[:, :-1]], dim=1)
        # compute attention distribution recursively
        # q[i] = (1 - p_choose_i[i]) * q[i-1] + previous_att[i]
        # att[i] = p_choose_i[i] * q[i]
        attention = []
        for ii in range(seq_len) :
            if ii == 0 :
                attention.append(previous_att[:, ii])
            else :
                attention.append((shifted_1mp_choose_i[:, ii]*attention[ii-1])+previous_att[:, ii])
        attention = torch.stack(attention, dim=1)
        attention = attention * p_choose_i
        pass
    elif mode == 'parallel' :
        raise NotImplementedError()
    elif mode == 'hard' :
        raise NotImplementedError()
    else :
        raise NotImplementedError()
    return attention
    pass

class BernoulliMonotonicAttention(BaseAttention) :
    def __init__(self, ctx_size, query_size, att_hid_size=256, act_fn=F.tanh) :
        super().__init__()
        raise ValueError("not tested implementation, please use other method")
        self.ctx_size = ctx_size
        self.query_size = query_size
        self.att_hid_size = att_hid_size
        self.act_fn = act_fn
        self.lin_in2proj = nn.Linear(ctx_size + query_size, att_hid_size)
        self.lin_proj2score = nn.Linear(att_hid_size, 1)
        self.out_features = self.ctx_size
        pass

    def forward(self, input) :
        ctx = input['ctx'] # batch x enc_len x enc_dim #
        query = input['query'] # batch x dec_dim #
        mask = input.get('mask', None) # batch x enc_len #
        batch, enc_len, enc_dim = ctx.size()
        if self.state is None :
            prev_att = torchauto(self).FloatTensor(batch, enc_len).zero_()
            prev_att[:, 0].add_(1)
            prev_att = Variable(prev_att)

            self.state = {'prev_att': prev_att}
        prev_att = self.state['prev_att']

        combined_input = torch.cat([ctx, query.unsqueeze(1).expand(batch, enc_len, self.query_size)], 2) # batch x enc_len x (enc_dim + dec_dim) #
        combined_input_2d = combined_input.view(batch * enc_len, -1)
        score_ctx = self.lin_proj2score(self.act_fn(self.lin_in2proj(combined_input_2d)))
        score_ctx = score_ctx.view(batch, enc_len) # batch x enc_len #
        if mask is not None :
            score_ctx = self.apply_mask(score_ctx, mask)
        p_ctx = _monotonic_probability_fn(score_ctx, prev_att, sigmoid_noise=True, mode='recursive')
        expected_ctx = self.calc_expected_context(p_ctx, ctx)

        # save state
        self.state['prev_att'] = p_ctx
        return expected_ctx, p_ctx

    def __call__(self, *input, **kwargs) :
        result = super().__call__(*input, **kwargs)
        expected_ctx, p_ctx = result
        return {"p_ctx":p_ctx, "expected_ctx":expected_ctx}
    pass
