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
from torchev.generator import generator_rnn, generator_attention, generator_act_fn
from torchev.custom import decoder
from torchev.utils.helper import torchauto
from torchev.utils.mask_util import generate_seq_mask
from torchev.base.model import BaseModel
from torchev.nn.modules import Conv2dEv
from torchev.custom.composite import ResidualBlock2D

# utilbox #
from utilbox.config_util import ConfigParser

"""
Deep Speaker : an End to End Neural Speaker Embedding System

Ref : https://arxiv.org/abs/1705.02304
"""

class DeepSpeakerCNN(BaseModel) :
    def __init__(self, in_size, out_emb_size=512, channel_size=[64, 128, 256, 512], 
            kernel_size=[(5, 5), (5, 5), (5, 5), (5, 5)], 
            stride=[(2, 2), (2, 2), (2, 2), (2, 2)],
            conv_fn_act='leaky_relu', pool_fn='avg',
            num_speaker=1024) :
        super().__init__()
        self.in_size = in_size
        self.out_emb_size = out_emb_size
        self.channel_size = channel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_fn_act = conv_fn_act
        
        # check model spec #
        assert len(channel_size) == len(kernel_size) == len(stride)
        assert pool_fn in ['avg', 'max'], "pool_fn must be 'avg' or 'max'"
        self.pool_fn = pool_fn
        self.num_layers = len(channel_size)

        # build model #
        
        # build conv layer #
        self.conv_lyr = nn.ModuleList()
        self.resblock_lyr = nn.ModuleList()
        curr_in_size = 1
        for ii in range(self.num_layers) :
            self.conv_lyr.append(Conv2dEv(curr_in_size, channel_size[ii], 
                kernel_size=tuple(kernel_size[ii]), stride=tuple(stride[ii])))
            self.resblock_lyr.append(ResidualBlock2D(out_channels=channel_size[ii]))
            curr_in_size = channel_size[ii]

        """ 
        WARNING :
        modified, should be flatten -> linear but it is hard to calculate the size exectly
        replaced with average pooling
        """
        in_emb_size = channel_size[-1]
        self.lin_emb_lyr = nn.Linear(in_emb_size, out_emb_size)

        # build softmax (for pretraining) #
        self.num_speaker = num_speaker
        self.lin_softmax_lyr = nn.Linear(out_emb_size, num_speaker)
        pass

    @property
    def config(self) :
        _config = super().config
        _config.update(dict(
            in_size=self.in_size,
            out_emb_size=self.out_emb_size,
            channel_size=self.channel_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            conv_fn_act=self.conv_fn_act,
            pool_fn=self.pool_fn,
            num_speaker=self.num_speaker
                    ))
        return _config

    def reset(self) :
        pass

    def forward(self, input, input_len=None) :
        batch, max_input_len, in_size = input.size()
        # convert to batch, channel, seq_len, n_dim 

        # apply masking #
        if input_len is not None :
            mask_input = Variable(generate_seq_mask(input_len, device=self, max_len=max_input_len).unsqueeze(-1))
            input = input * mask_input

        res = input.unsqueeze(1) 

        # apply conv 
        for ii in range(self.num_layers) :
            res = self.conv_lyr[ii](res)
            res = generator_act_fn(self.conv_fn_act)(res)
            res = self.resblock_lyr[ii](res)

        # res = [batch, out_channel, seq_len, n_dim] #
        # pool across seq_len #
        if self.pool_fn == 'avg' :
            res = F.avg_pool2d(res, kernel_size=[res.size(2), 1], stride=1)
        elif self.pool_fn == 'max' :
            res = F.max_pool2d(res, kernel_size=[res.size(2), 1], stride=1)
        else :
            raise ValueError("pool_fn {} is not implemented".format(self.pool_fn))

        # affine transform #
        # res = [batch, out_channel, 1, n_dim] #
        res = F.avg_pool2d(res, kernel_size=[1, res.size(-1)], stride=1)
        # res = [batch, out_channel, 1, 1] #
        res = res.squeeze(-1).squeeze(-1) # res = [batch, out_channel]
        res = self.lin_emb_lyr(res)
        # normalize to unit-norm #
        res = res / torch.norm(res, p=2, dim=1, keepdim=True)
        return res
    
    def forward_softmax(self, emb) :
        """
        Extra function to calculate speaker probability
        """
        return self.lin_softmax_lyr(emb)
    pass
