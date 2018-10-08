
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
from torchev.generator import generator_rnn, generator_attention
from torchev.custom import decoder
from torchev.utils.helper import torchauto, tensorauto

# utilbox #
from utilbox.config_util import ConfigParser


def SentDurRNN(Module) :
    def __init__(self, in_size, hid_size, num_layers, do=0.0, rnn_cfg={"type":"lstm", "bi":True}, summary='mean') :
        super(SentDurRNN, self).__init__()
        self.in_size = in_size
        self.hid_size = hid_size
        self.num_layers = num_layers
        self.do = do
        self.rnn_cfg = rnn_cfg
        self.summary = summary
        _rnn_cfg['args'] = [in_size, hid_size, num_layers, True, True, do, rnn_cfg.get('bi', False)]
        self.layers = generator_rnn(_rnn_cfg)
        self.regression = nn.Linear(hid_size*2 if self.layers.bidirectional else hid_size, 1)
        pass

    def forward(self, input, seq_len=None) :
        batch, max_seq_len, _ = input.size()
        if seq_len is None :
            seq_len = [max_seq_len] * batch
        res = input
        res = pack(res, seq_len, batch_first=True)
        res = self.layers(res)[0] # get h only #
        res,_ = unpack(res, batch_first=True)
        
        seq_len_var = Variable(torchauto(self).FloatTensor(seq_len).unsqueeze(1).expand(batch, res.size(2)))

        if self.summary == 'mean' :
            res = torch.sum(res, 1).squeeze(1) / seq_len_var
        elif self.summary == 'last' :
            _res = []
            for ii in range(batch) :
                if self.layers.bidirectional :
                    _last_fwd = res[ii, seq_len[ii]-1, 0:self.hid_size]
                    _last_bwd = res[ii, 0, self.hid_size*2:]
                    _res.append(torch.cat([_last_fwd, _last_bwd]))
                else :
                    _res.append(res[ii, seq_len[ii]-1, 0:self.hid_size])

            res = torch.stack(_res)
        return self.regression(res)

    def get_config(self) :
        # TODO
        return {'class':str(self.__class__),
                'in_size':self.in_size,
                'hid_size':self.hid_size,
                'num_layers':self.num_layers,
                'do':self.do,
                'rnn_cfg':self.rnn_cfg,
                'summary':self.summary,
                }
        pass 
    pass
