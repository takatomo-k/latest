import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.autograd import Variable

from ..generator import generator_rnn, generator_attention, generator_act_fn
from ..nn.modules.rnn import StatefulBaseCell
from ..utils.helper import is_cuda_module, apply_fn_variables
from utilbox.config_util import ConfigParser

class ContextGate(nn.Module):
    """
    Context gate is a decoder module that takes as input the previous word
    embedding, the current decoder state and the attention state, and
    produces a gate.
    The gate can be used to select the input from the target side context
    (decoder state), from the source context (attention state) or both.
    """
    def __init__(self, gate_type, source_size, target_size, output_size):
        super(ContextGate, self).__init__()
        input_size = source_size + target_size
        self.gate_lyr = nn.Linear(input_size, output_size, bias=True)
        self.source_proj_lyr = nn.Linear(source_size, output_size)
        self.target_proj_lyr = nn.Linear(target_size, output_size)
        assert gate_type in ['all', 'src', 'tgt']
        self.gate_type = gate_type

    def forward(self, source_var, target_var):
        assert source_var.shape[0] == target_var.shape[0]
        input_tensor = torch.cat((source_var, target_var), dim=1)
        z = F.sigmoid(self.gate_lyr(input_tensor))
        proj_source = self.source_proj_lyr(source_var)
        proj_target = self.target_proj_lyr(target_var)
        if self.gate_type == 'all' :
            return (z * proj_source) + (1.0 - z) * proj_target
        elif self.gate_type == 'src' :
            return (z * proj_source) + proj_target
        elif self.gate_type == 'tgt' :
            return proj_source + (1.0 - z) * proj_target
        else :
            raise NotImplementedError('gate_type {} is not defined'.format(self.gate_type))

class StandardDecoder(Module) :
    def __init__(self, att_cfg, ctx_size, in_size, 
            rnn_sizes=[512, 512], rnn_cfgs={'type':'stateful_lstmcell'}, 
            rnn_do=0.25, 
            ctx_proj_size=256, ctx_proj_fn='tanh',
            scale_src=1.0, scale_tgt=1.0,
            att_nth_layer=-1, input_feed=0,
            ctx_gate=None) :
        """
        ctx_proj_size : projection layer after context vector
        att_nth_layer : attach attention layer on n-th RNN layer
        input_feed : input feeding strategy (see Effective NMT)

        scale_src : scaling expected context vector before concat
        scale_tgt : scaling RNN hidden vector before concat
        """
        super().__init__()

        self.rnn_cfgs = rnn_cfgs
        self.rnn_sizes = rnn_sizes
        self.rnn_cfgs = rnn_cfgs
        self.rnn_do = ConfigParser.list_parser(rnn_do, len(rnn_sizes))
        self.ctx_proj_size = ctx_proj_size
        self.ctx_proj_fn = ctx_proj_fn
        assert input_feed >= 0 and input_feed < len(rnn_sizes)
        self.input_feed = input_feed
        if att_nth_layer >= 0 :
            self.att_nth_layer = att_nth_layer
        elif att_nth_layer < 0 :
            self.att_nth_layer = len(rnn_sizes) + att_nth_layer
            assert self.att_nth_layer >= 0
        self.scale_src = scale_src
        self.scale_tgt = scale_tgt

        assert ctx_gate in [None, 'all', 'src', 'tgt']
        self.ctx_gate = ctx_gate

        rnn_cfgs = ConfigParser.list_parser(rnn_cfgs, len(rnn_sizes))
        assert att_nth_layer >=1 or att_nth_layer <= -1
        self.stack_rnn_lyr = nn.ModuleList()
        prev_size = in_size
        for ii in range(len(rnn_sizes)) :
            prev_size += (ctx_proj_size if input_feed == ii else 0)
            rnn_cfg = rnn_cfgs[ii]
            rnn_cfg['args'] = [prev_size, rnn_sizes[ii]]
            _rnn_layer = generator_rnn(rnn_cfg)
            assert isinstance(_rnn_layer, StatefulBaseCell), "decoder can only use StatefulBaseCell layer"
            self.stack_rnn_lyr.append(_rnn_layer)
            prev_size = rnn_sizes[ii]
            if self.att_nth_layer == ii :
                # init attention #
                att_cfg['args'] = [ctx_size, rnn_sizes[self.att_nth_layer]]
                self.att_layer = generator_attention(att_cfg)
                if self.ctx_gate is None :
                    # if ctx_gate is None, we just need to create simple projection layer
                    self.ctx_proj_lyr = nn.Linear(rnn_sizes[ii]+self.att_layer.out_features, ctx_proj_size)
                prev_size = ctx_proj_size
            pass

        self.output_size = prev_size

        # additional : context gate (scaling information from source & target w/ non-linear proj)
        if self.ctx_gate is not None :
            self.ctx_gate_lyr = ContextGate(self.ctx_gate, 
                    self.att_layer.out_features, self.rnn_sizes[self.att_nth_layer], ctx_proj_size)
        
        # TODO : remove output_size argument #
        self.out_features = prev_size
        self.reset()
    
    @property
    def state(self) :
        _state = []
        for ii in range(len(self.stack_rnn_lyr)) :
            _state.append(self.stack_rnn_lyr[ii].state)
        _att_state = self.att_layer.state
        if self.input_feed is not None :
            return _state, self.ctx_proj_prev, _att_state
        else :
            return _state, None, _att_state
    
    @state.setter
    def state(self, value) :
        _state = value[0]
        assert len(_state) == len(self.stack_rnn_lyr)
        for ii in range(len(self.stack_rnn_lyr)) :
            self.stack_rnn_lyr[ii].state = _state[ii]
        self.ctx_proj_prev = value[1]
        self.att_layer.state = value[2]

    def detach_state(self) :
        """
        detach_state : detach current state from previous state for trunc BPTT
        call this function after calculate and update weight 
        """
        apply_fn_variables(self.state, lambda x : x.detach_())
        pass

    def set_ctx(self, ctx, ctx_mask=None) :
        self.ctx = ctx
        self.ctx_mask = ctx_mask
        if not isinstance(self.ctx_mask.data, (torch.FloatTensor, torch.cuda.FloatTensor)) :
            self.ctx_mask = self.ctx_mask.float()
        self.ctx_len = torch.sum(self.ctx_mask.data, dim=1).int().cpu().numpy().tolist()
    
    def reset(self) :
        for ii in range(len(self.stack_rnn_lyr)) :
            self.stack_rnn_lyr[ii].reset()
        self.ctx_proj_prev = None # additional for input-feeding module 
        self.att_layer.reset()

    def forward(self, dec_input, dec_mask=None) :
        batch = dec_input.size(0)
        x_below = dec_input
        if dec_mask is not None : # TODO masking helper 
            assert isinstance(dec_mask.data, (torch.FloatTensor, torch.cuda.FloatTensor))
        for ii in range(len(self.stack_rnn_lyr)) :
            if self.input_feed == ii :
                tt = torch.cuda if is_cuda_module(self) else torch
                if self.ctx_proj_prev is None : 
                    self.ctx_proj_prev = Variable(tt.FloatTensor(batch, self.ctx_proj_size).zero_())
                x_below = torch.cat([x_below, self.ctx_proj_prev], 1)
            res,_ = self.stack_rnn_lyr[ii](x_below)

            if dec_mask is not None :  
                res = res * dec_mask.unsqueeze(1).expand_as(res)

            if self.att_nth_layer != ii :
                res = F.dropout(res, self.rnn_do[ii], self.training)
            
            if self.att_nth_layer == ii :
                h_t = res
                att_res = self.att_layer({'ctx':self.ctx, 'mask':self.ctx_mask, 'query':h_t})
                
                if self.ctx_gate is None :
                    res = generator_act_fn(self.ctx_proj_fn)(self.ctx_proj_lyr(torch.cat(
                        [h_t * self.scale_tgt, att_res['expected_ctx'] * self.scale_src], 1)))
                else :
                    res = generator_act_fn(self.ctx_proj_fn)(
                            self.ctx_gate_lyr(att_res['expected_ctx'] * self.scale_src, h_t * self.scale_tgt)) 

                if dec_mask is not None : 
                    res = res * dec_mask.unsqueeze(1).expand_as(res)
                
                if self.input_feed is not None :
                    # no dropout for next ctx if stabler #
                    self.ctx_proj_prev = res

                res = F.dropout(res, self.rnn_do[ii], self.training)
                pass
            x_below = res
            pass
        return x_below, att_res

    def __call__(self, *input, **kwargs) :
        result = super(StandardDecoder, self).__call__(*input, **kwargs)
        x_below, att_res = result
        return {"dec_output":x_below, "att_output":att_res}
        
    pass

