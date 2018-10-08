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
from torch.nn import init

# torchev #
from torchev.generator import generator_rnn, generator_attention, generator_act_fn, generator_act_module
from torchev.custom import decoder
from torchev.custom.composite import CBHG1d, MultiscaleConv1d
from torchev.custom.composite import MultiscaleConv1d
from torchev.utils.helper import torchauto, tensorauto
from torchev.utils.mask_util import generate_seq_mask
from torchev.nn import Conv1dEv, Conv2dEv

# utilbox #
from utilbox.config_util import ConfigParser

class TacotronV1Inverter(nn.Module) :

    """
    Tacotron 1 inverter Mel-Spectrogram -> Linear Spectrogram with CBHG module
    """

    def __init__(self, in_size, out_size, 
            projection_size=[512], projection_fn='LeakyReLU', projection_do=0.0,
            cbhg_cfg={}
            ) :
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.projection_size = projection_size
        self.projection_fn = projection_fn
        self.projection_do = ConfigParser.list_parser(projection_do, n=len(projection_size))

        self.inverter_lyr = CBHG1d(in_size, 
                conv_proj_filter=[256, in_size], **cbhg_cfg)
        _tmp = []
        prev_size = self.inverter_lyr.out_features
        for ii in range(len(projection_size)) :
            _tmp.append(nn.Linear(prev_size, self.projection_size[ii]))
            _tmp.append(generator_act_module(self.projection_fn))
            _tmp.append(nn.Dropout(p=self.projection_do[ii]))
            prev_size = self.projection_size[ii]
            pass
        _tmp.append(nn.Linear(prev_size, out_size))
        self.projection_lyr = nn.Sequential(*_tmp)
        pass

    def get_config(self) :
        return {'class':str(self.__class__),
                'in_size':self.in_size,
                'out_size':self.out_size,
                'projection_size':self.projection_size,
                'projection_fn':self.projection_fn,
                'projection_do':self.projection_do,
                }
    def reset(self) :
        pass

    def forward(self, input, seq_len=None) :
        batch, max_seq_len, _ = input.size()
        res = self.inverter_lyr(input, seq_len)
        res = res.contiguous().view(batch * max_seq_len, -1)
        res = self.projection_lyr(res)
        res = res.contiguous().view(batch, max_seq_len, -1)
        return res

class TacotronMultiConvInverter(nn.Module) :
    def __init__(self, in_size, out_size, 
            conv_bank_k=8, conv_fn_act='leaky_relu', conv_bank_filter=[128, 128],
            conv_do=0.25) :
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.conv_bank_filter = conv_bank_filter 
        self.conv_bank_k = ConfigParser.list_parser(conv_bank_k, len(self.conv_bank_filter))
        self.conv_fn_act = conv_fn_act
        self.conv_do = ConfigParser.list_parser(conv_do, len(self.conv_bank_filter))
        
        self.conv_bank_lyrs = nn.ModuleList()
        prev_size = in_size
        for ii in range(len(conv_bank_filter)) :
            self.conv_bank_lyrs.append(
                    MultiscaleConv1d(prev_size, out_channels=conv_bank_filter[ii], 
                        kernel_sizes=list(range(1, self.conv_bank_k[ii]+1)), padding='same'))
            prev_size = self.conv_bank_lyrs[-1].out_channels
            pass
        self.lin_pred_lyr = nn.Linear(prev_size, out_size)
        pass

    def get_config(self) :
        return {'class':str(self.__class__),
                'in_size':self.in_size,
                'out_size':self.out_size,
                'conv_bank_filter':self.conv_bank_filter,
                'conv_bank_k':self.conv_bank_k,
                'conv_fn_act':self.conv_fn_act,
                'conv_do':self.conv_do
                }

    def forward(self, input, seq_len=None) :
        if seq_len is not None :
            mask_input = Variable(generate_seq_mask(seq_len=seq_len, device=self).unsqueeze(-1)) # batch x seq_len x 1 #
            mask_input_conv = mask_input.transpose(1, 2) # batch x 1 x seq_len
        else :
            mask_input = None
        
        if mask_input is not None :
            input = input * mask_input

        res = input
        res = res.transpose(1, 2)
        for ii in range(len(self.conv_bank_lyrs)) :
            res = self.conv_bank_lyrs[ii](res)
            res = generator_act_fn(self.conv_fn_act)(res)
            if self.conv_do[ii] > 0.0 :
                res = F.dropout(res, p=self.conv_do[ii], training=self.training)
            if mask_input is not None :
                res = res * mask_input_conv
        res = res.transpose(1, 2) # batch x seq_len x ndim
        # apply linear layer #
        res = self.lin_pred_lyr(res)
        if mask_input is not None :
            res = res * mask_input
        return res

    pass
