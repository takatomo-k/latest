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
from torchev.custom import decoder
from torchev.utils.helper import torchauto
from torchev.utils.mask_util import generate_seq_mask
# utilbox #
from utilbox.config_util import ConfigParser

class GeneratorStaticRNN(nn.Module) :
    """
    sequence 1-to-1 mapping 

    Ver 1
    input 
    -> residual conv (N layer, pad same)
    -> RNN (LSTM or GRU) (N layer, pad same, uni/bi)
    -> residual conv (N layer, pad same)
    -> linear layer (1 layer)
    """
    def __init__(self, enc_in_size, dec_out_size,
            enc_conv_ch=[256, 256], enc_conv_ksize=[4, 4], enc_conv_do=[0.0, 0.0], 
            enc_conv_act='leaky_relu',
            rnn_cfgs={'type':'lstm', 'bi':True}, rnn_sizes=[256, 256],
            dec_conv_ch=[256, 256], dec_conv_ksize=[4, 4], dec_conv_do=[0.0, 0.0], 
            dec_conv_act='leaky_relu',
            ) :
        super().__init__()
        self.enc_in_size = enc_in_size
        self.dec_out_size = dec_out_size

        self.enc_conv_ch = enc_conv_ch
        self.enc_conv_ksize = enc_conv_ksize
        self.enc_conv_do = enc_conv_do
        self.enc_conv_act = enc_conv_act

        self.rnn_sizes = rnn_sizes
        self.rnn_cfgs = rnn_cfgs

        self.dec_conv_ch = dec_conv_ch
        self.dec_conv_ksize = dec_conv_ksize
        self.dec_conv_do = dec_conv_do
        self.dec_conv_act = dec_conv_act

        # init enc #
        prev_ch = enc_in_size
        self.enc_conv_lyr = nn.ModuleList()
        for ii in range(len(self.enc_conv_ksize)) :
            self.enc_conv_lyr.append(nnev.Conv1dEv(prev_ch, 
                self.enc_conv_ch[ii], self.enc_conv_ksize[ii], padding='same'))
            prev_ch = enc_conv_ch[ii]
            pass
        
        # init rnn (middle part) #
        self.rnn_lyr = nn.ModuleList()
        _rnn_cfgs = ConfigParser.list_parser(rnn_cfgs, len(rnn_sizes))
        prev_size = enc_conv_ch[-1]
        for ii in range(len(rnn_cfgs)) :
            _rnn_cfg = {}
            _rnn_cfg['type'] = _rnn_cfgs[ii]['type']
            _rnn_cfg['args'] = [prev_size, self.rnn_sizes[ii], 1, True, True, 0, _rnn_cfgs[ii]['bi']]
            self.rnn_lyr.append(generator_rnn(_rnn_cfg))
            prev_size = rnn_sizes[ii] * (2 if _rnn_cfgs[ii]['bi'] else 1)
            pass

        # init dec #
        prev_ch = prev_size
        self.dec_conv_lyr = nn.ModuleList()
        for ii in range(len(self.dec_conv_ksize)) :
            self.dec_conv_lyr.append(nnev.Conv1dEv(prev_ch, 
                self.dec_conv_ch[ii], self.dec_conv_ksize[ii], padding='same'))
            prev_ch = self.dec_conv_ch[ii]
            pass
        
        # init regression #
        self.dec_core_reg_lyr = nn.Linear(prev_ch, dec_out_size)

    def get_config(self) :
        return {'class':str(self.__class__),
            'enc_in_size':self.enc_in_size,
            'dec_out_size':self.dec_out_size,
            'enc_conv_ch':self.enc_conv_ch,
            'enc_conv_ksize':self.enc_conv_ksize,
            'enc_conv_do':self.enc_conv_do,
            'enc_conv_act':self.enc_conv_act,
            'rnn_sizes':self.rnn_sizes,
            'rnn_cfgs':self.rnn_cfgs,
            'dec_conv_ch':self.dec_conv_ch,
            'dec_conv_ksize':self.dec_conv_ksize,
            'dec_conv_do':self.dec_conv_do,
            'dec_conv_act':self.dec_conv_act,
            }
        pass

    @property
    def state(self) :
        return None

    @state.setter
    def state(self, value) :
        pass

    def transcode(self, input, src_len=None) :
        """
        input : (batch x max_src_len x in_size)
        mask : (batch x max_src_len)
        """
        batch, max_src_len, in_size = input.size()
        
        if src_len is None :
            src_len = [max_src_len] * batch
        res = input.transpose(1, 2) # batch x ndim x seqlen 

        # fwd enc conv #
        for ii in range(len(self.enc_conv_lyr)) :
            res = self.enc_conv_lyr[ii](res)
            res = generator_act_fn(self.enc_conv_act)(res)
            res = F.dropout(res, self.enc_conv_do[ii])
            pass

        # fwd rnn #
        res = res.transpose(1, 2) # batch x seqlen x ndim
        for ii in range(len(self.rnn_lyr)) :
            res = pack(res, src_len, batch_first=True)
            res = self.rnn_lyr[ii](res)[0] # get h only #
            res,_ = unpack(res, batch_first=True)
            # TODO dropout # 
        
        # fwd dec conv #
        res = res.transpose(1, 2) # batch x ndim x seqlen
        
        for ii in range(len(self.dec_conv_lyr)) :
            res = self.dec_conv_lyr[ii](res)
            res = generator_act_fn(self.dec_conv_act)(res)
            res = F.dropout(res, self.dec_conv_do[ii], training=self.training)
            pass
        res = res.transpose(1, 2) # batch x seqlen x channel
        # fwd core regression #
        res = res.contiguous().view(batch * max_src_len, res.size(-1))
        res = self.dec_core_reg_lyr(res)
        res = res.view(batch, max_src_len, res.size(-1))
        return res
    pass
