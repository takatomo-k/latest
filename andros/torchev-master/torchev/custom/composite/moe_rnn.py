import math
import torch

from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from ...nn.modules.rnn import StatefulBaseCell
from ...utils.helper import torchauto, tensorauto

class StatefulMOEGRUCell(StatefulBaseCell) :
    def __init__(self, input_size, hidden_size, n_experts=3, bias=True) :
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_experts = n_experts

        self.bias = bias
        
        self.weight_ih = nn.ModuleList([nn.Linear(input_size, hidden_size*3, bias=bias) for _ in range(n_experts)])
        self.weight_hh = nn.ModuleList([nn.Linear(hidden_size, hidden_size*3, bias=bias) for _ in range(n_experts)])
        self.gating_ih = nn.Linear(input_size, n_experts)
        self.gating_hh = nn.Linear(hidden_size, n_experts)
        pass

    def forward(self, input) :
        batch = input.size(0)
        if self.state is None :
            h0 = Variable(torchauto(self).FloatTensor(batch, self.hidden_size).zero_())
        else :
            h0 = self.state

        # calculate MoE #
        gate_ih_t = F.softmax(self.gating_ih(input), dim=1)
        gate_hh_t = F.softmax(self.gating_hh(h0), dim=1)
        list_pre_ih, list_pre_hh = [], []

        for ii in range(self.n_experts) :
            list_pre_ih.append(self.weight_ih[ii](input))
            list_pre_hh.append(self.weight_hh[ii](h0))
        list_pre_ih = torch.stack(list_pre_ih, dim=1)
        list_pre_hh = torch.stack(list_pre_hh, dim=1)

        pre_ih = gate_ih_t.unsqueeze(-1) * list_pre_ih
        pre_hh = gate_hh_t.unsqueeze(-1) * list_pre_hh
        pre_ih = pre_ih.sum(dim=1)
        pre_hh = pre_hh.sum(dim=1)
        # end #

        pre_rih, pre_zih, pre_nih = torch.split(pre_ih, self.hidden_size, dim=1)
        pre_rhh, pre_zhh, pre_nhh = torch.split(pre_hh, self.hidden_size, dim=1)
        r_t = F.sigmoid(pre_rih + pre_rhh)
        z_t = F.sigmoid(pre_zih + pre_zhh)
        c_t = F.tanh(pre_nih + r_t * (pre_nhh))
        h_t = (1-z_t) * c_t + (z_t * h0)
        self.state = h_t
        return h_t
        pass
    pass    


