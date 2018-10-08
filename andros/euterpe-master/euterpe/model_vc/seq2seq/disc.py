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

class DiscriminatorCNN(nn.Module) :
    def __init__(self, in_size,
            conv_ch=[64, 128, 128], conv_ksize=[40, 5, 3], conv_stride=[2, 1, 1],
            conv_act='leaky_relu', conv_do=0.0
            ) :
        super().__init__()
        
        self.conv_ch = conv_ch
        self.conv_ksize = conv_ksize
        self.conv_stride = conv_stride
        self.conv_act = conv_act 
        self.conv_do = ConfigParser.list_parser(conv_do, len(self.conv_ch))
        
        self.conv_lyr = nn.ModuleList()
        prev_ch = in_size
        for ii in range(len(self.conv_ksize)) :
            self.conv_lyr.append(nnev.Conv1dEv(prev_ch, 
                self.conv_ch[ii], self.conv_ksize[ii], 
                stride=self.conv_stride[ii], padding='valid'))
            prev_ch = self.conv_ch[ii]
            pass
        self.disc_final_layer = nn.Linear(prev_ch, 1)
        pass

    def forward(self, input) :
        """
        input : (batch x max_src_len x in_size)
        mask : (batch x max_src_len)
        """
        batch, max_src_len, in_size = input.size()
        
        res = input.transpose(1, 2) # batch x ndim x seqlen 

        for ii in range(len(self.conv_ksize)) :
            res = self.conv_lyr[ii](res)
            res = generator_act_fn(self.conv_act)(res)
            res = F.dropout(res, self.conv_do[ii], training=self.training)
            pass

        res = res.transpose(1, 2) # batch x seqlen x ndim #
        batch, curr_len, curr_ndim = res.size()
        res = res.contiguous().view(batch*curr_len, curr_ndim)
        res = self.disc_final_layer(res)
        res = res.view(batch, curr_len) # change to 2 dim again #
        return res
        pass
    pass
