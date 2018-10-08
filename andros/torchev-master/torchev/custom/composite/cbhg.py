import torch
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch import nn
from torch.nn import init

from ...utils.helper import torchauto, tensorauto
from ...utils.mask_util import generate_seq_mask
from ...generator import generator_act_fn, generator_rnn
from .multiscale_conv import MultiscaleConv1d
from ...nn.modules.pool import MaxPool1dEv
from ...nn.modules.conv import Conv1dEv
from .highway import HighwayFNN

from utilbox.config_util import ConfigParser
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack


class CBHG1d(Module) :
    """
    Ref : 
        Tacotron : Towards End-to-End Speech Sythesis
    CBHG composed based on 3 modules : 
        Convolution 1D - Highway Network - Bidirectional {GRU, LSTM}

    This is the 1st module 

    Input format : ( batch x time x input dim )
    Output format : ( batch x time x sum(conv_channels) )
    """
    def __init__(self, in_size, conv_bank_k=8, conv_bank_act='leaky_relu', conv_bank_filter=128,
            pool_size=2,
            conv_proj_k=[3, 3], conv_proj_filter=[128, 128], conv_proj_act=['leaky_relu', 'none'],
            highway_size=128, highway_lyr=4, highway_act='leaky_relu',
            rnn_cfgs={'type':'gru', 'bi':True}, rnn_sizes=[128],
            use_bn=True
            ) :
        super(CBHG1d, self).__init__()
        # conv bank multiscale #
        self.conv_bank_lyr = MultiscaleConv1d(in_size, conv_bank_filter, kernel_sizes=list(range(1, conv_bank_k+1)), padding='same')
        if use_bn :
            self.conv_bank_bn = nn.BatchNorm1d(self.conv_bank_lyr.out_channels)
        self.conv_bank_act = conv_bank_act
        self.pool_lyr = MaxPool1dEv(pool_size, stride=1, padding='same')
        self.conv_proj_lyr = nn.ModuleList()
        if use_bn :
            self.conv_proj_bn = nn.ModuleList()
        prev_filter = self.conv_bank_lyr.out_channels
        for ii in range(len(conv_proj_k)) :
            self.conv_proj_lyr.append(Conv1dEv(prev_filter, conv_proj_filter[ii], kernel_size=conv_proj_k[ii], padding='same'))
            if use_bn :
                self.conv_proj_bn.append(nn.BatchNorm1d(conv_proj_filter[ii]))
            prev_filter = conv_proj_filter[ii]
        self.conv_proj_act = conv_proj_act 
        assert prev_filter == in_size
        self.pre_highway_lyr = nn.Linear(prev_filter, highway_size)
        self.highway_lyr = HighwayFNN(highway_size, highway_lyr, fn_act=generator_act_fn(highway_act))
        self.highway_act = highway_act
        self.use_bn = use_bn

        self.rnn_lyr = nn.ModuleList()
        rnn_cfgs = ConfigParser.list_parser(rnn_cfgs, len(rnn_sizes))
        prev_size = highway_size
        for ii in range(len(rnn_sizes)) :
            _rnn_cfg = {}
            _rnn_cfg['type'] = rnn_cfgs[ii]['type']
            _rnn_cfg['args'] = [prev_size, rnn_sizes[ii], 1, True, True, 0, rnn_cfgs[ii]['bi']]
            self.rnn_lyr.append(generator_rnn(_rnn_cfg))
            prev_size = rnn_sizes[ii] * (2 if rnn_cfgs[ii]['bi'] else 1)
        self.output_size = prev_size
        self.out_features = prev_size
        pass

    def forward(self, input, seq_len=None) :

        batch, max_seq_len, ndim = input.size()
        # create mask #
        if seq_len is not None :
            mask_input = generate_seq_mask(seq_len=seq_len, device=self, max_len=max_seq_len)
            mask_input = Variable(mask_input.unsqueeze(-1)) # batch x seq_len x 1 #
            mask_input_t12 = mask_input.transpose(1, 2)
        else :
            mask_input = None

        if mask_input is not None :
            input = input * mask_input

        seq_len = [max_seq_len for _ in range(batch)] if seq_len is None else seq_len

        res_ori = input.transpose(1, 2) # saved for residual connection #
        res = input.transpose(1, 2) # batch x ndim x seq_len #

        # apply multiscale conv #
        res = self.conv_bank_lyr(res)

        res = generator_act_fn(self.conv_bank_act)(res)

        if mask_input is not None :
            res = res * mask_input_t12

        if self.use_bn :
            res = self.conv_bank_bn(res)
            if mask_input is not None :
                res = res * mask_input_t12

        # apply pooling #
        res = self.pool_lyr(res)
        if mask_input is not None :
            res = res * mask_input_t12

        # apply conv proj #
        for ii in range(len(self.conv_proj_lyr)) :
            res = self.conv_proj_lyr[ii](res)
            res = generator_act_fn(self.conv_proj_act[ii])(res)
            if mask_input is not None :
                res = res * mask_input_t12
            if self.use_bn :
                res = self.conv_proj_bn[ii](res)
                if mask_input is not None :
                    res = res * mask_input_t12

        # apply residual connection #
        assert list(res.size()) == list(res_ori.size())
        res = res + res_ori
        if mask_input is not None :
            res = res * mask_input_t12
        # change shape for feedforward #
        res = res.transpose(1, 2) # batch x seq_len x ndim #
        res = res.contiguous().view(batch * max_seq_len, -1) # (batch * seq_len) x ndim #
        # apply pre highway #
        res = generator_act_fn(self.highway_act)(self.pre_highway_lyr(res))
        # apply highway #
        res = self.highway_lyr(res)
        # apply rnn #
        res = res.contiguous().view(batch, max_seq_len, -1) # batch x seq_len x ndim #
        if mask_input is not None :
            res = res * mask_input

        for ii in range(len(self.rnn_lyr)) :
            res = pack(res, seq_len, batch_first=True)
            res = self.rnn_lyr[ii](res)[0]
            res, _ = unpack(res, batch_first=True)
            pass
        return res
