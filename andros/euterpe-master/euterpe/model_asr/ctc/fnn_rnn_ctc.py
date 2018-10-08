
import sys
import time

import numpy as np
import json

# pytorch #
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

from torch.autograd import Variable

# torchev #
from torchev.generator import generator_rnn, generator_act_fn

# utilbox #
from utilbox.config_util import ConfigParser

class FNN_RNN_CTC(nn.Module) :
    def __init__(self, in_size, n_class, fnn_sizes, rnn_sizes, 
            fnn_do=0.0, rnn_do=0.0, 
            downsampling=None,
            fnn_act='tanh', rnn_cfgs={"type":"lstm", "bi":True},
            use_pack=False,
            ) :
        super().__init__()

        self.in_size = in_size
        self.n_class = n_class
        self.fnn_sizes = fnn_sizes
        self.rnn_sizes = rnn_sizes
        assert len(set(self.fnn_sizes)) == 1
        assert len(set(self.rnn_sizes)) == 1
        self.downsampling = downsampling
        self.fnn_do = fnn_do
        self.rnn_do = rnn_do
        self.fnn_act = fnn_act
        self.rnn_cfgs = rnn_cfgs
        self.use_pack = use_pack

        # modules #
        self.fnn_lyr = nn.ModuleList()
        prev_size = in_size
        for ii in range(len(fnn_sizes)) :
            self.fnn_lyr.append(nn.Linear(prev_size, fnn_sizes[ii]))
            prev_size = fnn_sizes[ii]
        
        _rnn_cfg = rnn_cfgs
        _rnn_cfg['type'] = _rnn_cfg['type']
        _rnn_cfg['args'] = [prev_size, rnn_sizes[0], len(rnn_sizes), True, True, self.rnn_do, _rnn_cfg['bi']]
        self.rnn_lyr = generator_rnn(_rnn_cfg)
             
        prev_size = rnn_sizes[-1] * (2 if _rnn_cfg['bi'] else 1)

        self.pre_softmax = nn.Linear(prev_size, n_class)
        pass

    def forward(self, x, src_len=None) :
        """
        x : (batch x seq x ndim) 
        src_len : (batch)
        """
        batchsize, seqlen, ndim = x.size()
        if src_len is None :
            src_len = [seqlen] * batchsize
        
        ### FNN ###
        # convert shape for FNN #
        res = x.contiguous().view(seqlen * batchsize, ndim)
        
        for ii in range(len(self.fnn_sizes)) :
            res = generator_act_fn(self.fnn_act)(self.fnn_lyr[ii](res))
            res = F.dropout(res, self.fnn_do, training=self.training)
         
        ### RNN ###
        # convert shape for RNN #
        res = res.view(batchsize, seqlen,  -1)
        if self.use_pack :
            res = pack(res, src_len, batch_first=True)
        res = self.rnn_lyr(res)[0] 
        if self.use_pack :
            res,_ = unpack(res, batch_first=True)

        ### PRE SOFTMAX ###
        batchsize, seqlen_final, ndim_final = res.size()
        res = res.contiguous().view(seqlen_final * batchsize, ndim_final)

        res = self.pre_softmax(res)
        res = res.view(batchsize, seqlen_final, -1) 
        res = res.transpose(1, 0)
        return res, Variable(torch.IntTensor(src_len))
    
    def get_config(self) :
        return {'class':str(self.__class__),
                'in_size':self.in_size,
                'n_class':self.n_class,
                'fnn_sizes':self.fnn_sizes,
                'rnn_sizes':self.rnn_sizes,
                'fnn_do':self.fnn_do,
                'rnn_do':self.rnn_do,
                'downsampling':self.downsampling,
                'fnn_act':self.fnn_act,
                'rnn_cfgs':self.rnn_cfgs,
                'use_pack':self.use_pack
                }

    pass
