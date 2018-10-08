import torch

from torch import nn
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack


from ..generator import generator_rnn, generator_attention
from ..nn.modules.rnn import StatefulBaseCell
from ..utils.helper import is_cuda_module, apply_fn_variables
from ..utils.seq_util import downsampling
from utilbox.config_util import ConfigParser

class StandardRNNEncoder(Module) :
    def __init__(self, in_size, rnn_sizes=[512, 512], rnn_cfgs={'type':'lstm', 'bi':True},
            do=0.25, downsampling={'type':'none'}) :
        super().__init__() 

        self.in_size = in_size
        self.rnn_sizes = rnn_sizes
        self.rnn_cfgs = rnn_cfgs

        self.do = ConfigParser.list_parser(do, len(self.rnn_sizes))
        self.downsampling = ConfigParser.list_parser(downsampling, len(self.rnn_sizes))
        
        # init rnn #
        self.rnn_lyr = nn.ModuleList()
        _rnn_cfgs = ConfigParser.list_parser(self.rnn_cfgs, len(rnn_sizes))
        prev_size = in_size
        for ii in range(len(self.rnn_sizes)) :
            _rnn_cfg = {'type':_rnn_cfgs[ii]['type'], 'args':[prev_size, rnn_sizes[ii], 1, True, True, 0, _rnn_cfgs[ii].get('bi', False)]}
            self.rnn_lyr.append(generator_rnn(_rnn_cfg))
            prev_size = self.rnn_lyr[ii].hidden_size * (2 if self.rnn_lyr[ii].bidirectional else 1)
            pass
        self.output_size = prev_size
        self.out_features = self.output_size
        pass

    def forward(self, input, input_len=None) :
        assert input.dim() == 3, "input must be 3-dim tensor (batch x seq_len x ndim)"
        batch, max_in_len, in_size = input.size()

        if input_len is None :
            input_len = [max_in_len] * batch

        res = input
        res_len = input_len
        for ii in range(len(self.rnn_lyr)) :
            res = pack(res, res_len, batch_first=True) 
            res = self.rnn_lyr[ii](res)[0] # get h only #
            res,_ = unpack(res, batch_first=True)
            res = F.dropout(res, self.do[ii], self.training)
            res, res_len = downsampling(cfg=self.downsampling[ii], 
                    mat=res, mat_len=res_len)
            pass
        return res, res_len

    def __call__(self, *input, **kwargs) :
        result = super().__call__(*input, **kwargs)
        res, res_len = result
        return {'enc_output':res, 'enc_len':res_len}
    
    def get_config(self) :
        return {
                'class':str(self.__class__),
                'in_size':self.in_size,
                'rnn_sizes':self.rnn_sizes,
                'rnn_cfgs':self.rnn_cfgs,
                'do':self.do,
                'downsampling':self.downsampling
                }
    pass

