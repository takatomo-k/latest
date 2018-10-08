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

class Mel2SpecFNN(nn.Module) :
    def __init__(self, in_size, out_size, hid_sizes=[512, 512], act_fn='leaky_relu', do=0.0) :
        super(Mel2SpecFNN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hid_sizes = hid_sizes
        self.act_fn = act_fn
        self.do = ConfigParser.list_parser(do, len(self.hid_sizes))

        _fnns = []
        prev_size = in_size
        for ii in range(len(self.hid_sizes)) :
            _fnns.append(nn.Linear(prev_size, hid_sizes[ii]))
            prev_size = hid_sizes[ii]

        self.layers = nn.ModuleList(_fnns)
        self.proj = nn.Linear(prev_size, out_size)
        pass

    def forward(self, input) :
        batch, seq_len, _ = input.size()
        res = input.contiguous().view(batch * seq_len, self.in_size)
        for ii in range(len(self.hid_sizes)) :
            res = self.layers[ii](res)
            res = getattr(F, self.act_fn)(res)
            res = F.dropout(res, self.do[ii], self.training)
        res = self.proj(res)
        res = res.view(batch, seq_len, self.out_size)
        return res


    def get_config(self) :
        # TODO
        return {'class':str(self.__class__),
                'in_size':self.in_size,
                'out_size':self.out_size,
                'hid_sizes':self.hid_sizes,
                'act_fn':self.act_fn,
                'do':self.do
                }
                
    pass
