import sys
import time
import re

import numpy as np
import json

# pytorch #
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

# torchev #
import torchev.nn as nnev
from torchev.generator import generator_rnn, generator_attention, generator_act_fn
from torchev.custom import decoder, encoder
from torchev.utils.helper import torchauto
from torchev.utils.mask_util import generate_seq_mask
# utilbox #
from utilbox.config_util import ConfigParser

class GeneratorSeq2SeqRNN(nn.Module) :
    """
    sequence N-to-M mapping

    Ver 1.
    Enc RNN - Dec RNN
    """

    def __init__(self, enc_in_size, dec_in_size, dec_out_size,
            dec_rnn_sizes=[512, 512],
            dec_rnn_cfgs={'type':'lstm'}, dec_rnn_do=0.25,
            dec_cfg={'type':'standard_decoder'}, 
            att_cfg={'type':'mlp'}) :
        super().__init__()
        self.enc_in_size = enc_in_size
        self.dec_in_size = dec_in_size
        self.dec_out_size = dec_out_size
       
        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_cfgs = dec_rnn_cfgs
        self.dec_rnn_do = ConfigParser.list_parser(dec_rnn_do, len(dec_rnn_sizes))
        self.dec_cfg = dec_cfg
        self.att_cfg = att_cfg

        # init encoder #
        self.enc_lyr = encoder.StandardRNNEncoder(enc_in_size, do=0.0, 
            downsampling={'type':'last', 'step':2})
        
        # init decoder #
        _dec_rnn_cfgs = ConfigParser.list_parser(dec_rnn_cfgs, len(dec_rnn_sizes))
        for ii in range(len(dec_rnn_sizes)) :
            _type = _dec_rnn_cfgs[ii]['type']
            if re.match('stateful.*cell', _type) is None :
                _dec_rnn_cfgs[ii]['type'] = 'stateful_{}cell'.format(_type)

        prev_size = dec_in_size
        self.dec_lyr = decoder.StandardDecoder(att_cfg, self.enc_lyr.out_features,
                prev_size, dec_rnn_sizes, _dec_rnn_cfgs, self.dec_rnn_do)

        # init decoder regression #
        self.dec_core_reg_lyr = nn.Linear(self.dec_lyr.out_features, dec_out_size)
        pass

    def get_config(self) :
        return {'class':str(self.__class__),
            'enc_in_size':self.enc_in_size,
            'dec_in_size':self.dec_in_size,
            'dec_out_size':self.dec_out_size,
            'dec_rnn_sizes':self.dec_rnn_sizes,
            'dec_rnn_cfgs':self.dec_rnn_cfgs,
            'dec_rnn_do':self.dec_rnn_do,
            'dec_cfg':self.dec_cfg,
            'att_cfg':self.att_cfg
            }
        pass

    @property
    def state(self) :
        return None

    @state.setter
    def state(self, value) :
        pass

    def encode(self, input, src_len=None) :
        """
        input : (batch x max_src_len x in_size)
        mask : (batch x max_src_len)
        """
        batch, max_src_len, in_size = input.size()
        if src_len is None :
            src_len = [max_src_len] * batch
        res = self.enc_lyr(input, src_len) 
        ctx = res['enc_output']
        ctx_len = res['enc_len']
        ctx_mask = None
        if src_len is not None :
            ctx_mask = Variable(generate_seq_mask(
                seq_len=src_len, device=self, max_len=ctx.size(1)))
        self.dec_lyr.set_ctx(ctx, ctx_mask)
        pass
    
    def reset(self) :
        self.dec_lyr.reset()

    def decode(self, y_tm1, mask=None) :
        """
        decode y_t given y_tm1
        return y_t and related information from decoder attention
        """
        assert y_tm1.dim() == 2, 'batch x out_ndim only'
        res = y_tm1
        res = self.dec_lyr(res, mask)
        return self.dec_core_reg_lyr(res['dec_output']), res
        pass
    pass
