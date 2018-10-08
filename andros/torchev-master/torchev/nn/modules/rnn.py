import math
import torch

from torch.nn import Module
from torch.nn import Parameter
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from .activation import hard_sigmoid, st_bernoulli
from ...math.function import where
from ...utils.helper import torchauto

class HMLSTM(Module) :
    """
    Hierarchical Multiscale LSTM (Normal LSTM)
    """
    def __init__(input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False) :
        super(HMLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers; assert num_layers == 1, "currently only support 1 layer"
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1; assert num_directions == 1, "currently only support 1 direction LSTM"
        
        self._all_weights = []
        ### INIT PARAM ###
        for layer in range(num_layers) :
            for direction in range(num_directions) :
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                
                gate_size = 4 * hidden_size + 1 # +1 for z[t] #
                
                W_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                setattr(self, weights[0], w_ih)
                setattr(self, weights[1], w_hh)
                if bias :
                    setattr(self, weights[2], b_ih)
                    setattr(self, weights[3], b_hh)
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]
        self.reset_parameters()
        pass
    def reset_parameters(self) :
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, input, hx=None) :
        """
        
        """
        is_packed = isinstance(input, PackedSequence)
        if is_packed :
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else :
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None :
            num_directions = 2 if self.bidirectional else 1
            hx = Variable(input.data.new(self.num_layers * 
                                        num_directions, 
                                        max_batch_size, 
                                        self.hidden_size).zero_())
            # hx = (h0, c0) #
            hx = (hx, hx)


        
        pass
    pass

class HMLSTMCell(Module) :
    def __init__(self, input_size, hidden_size, upper_hidden_size=None, bias=True) :
        super(HMLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        upper_hidden_size = hidden_size if upper_hidden_size is None else upper_hidden_size
        self.upper_hidden_size = upper_hidden_size

        gate_size = hidden_size * 4 + 1
        
        self.w_rec = Parameter(torch.Tensor(gate_size, hidden_size))
        self.w_topdown = Parameter(torch.Tensor(gate_size, upper_hidden_size))
        self.w_bottomup = Parameter(torch.Tensor(gate_size, input_size))
        if bias :
            self.bias = Parameter(torch.Tensor(gate_size))
        else :
            self.register_parameter('bias', None)
        self.reset_parameters()
        pass

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden) :
        return HMLSTMBackend.forward(input, hidden, self.w_rec, self.w_topdown, self.w_bottomup, self.bias)
        pass
    pass

class HMLSTMBackend(Module) :
    @staticmethod
    def forward(input, hidden, w_rec, w_topdown, w_bottomup, bias=None, fn_z=F.sigmoid, stochastic=False) :
        """
        input: (batch, input_size) 
        """
        # LSTM dependency (Eq. 1) #
        h_lm1_t = input
        h_l_tm1, h_lp1_tm1, c_tm1, z_l_tm1, z_lm1_t = hidden
        batch_sizes, hidden_size = h_l_tm1.size()
        gates_size = 4 * hidden_size + 1
        # broadcast z #
        z_l_tm1_broadcast = z_l_tm1.unsqueeze(1).expand(batch_sizes, gates_size)
        z_lm1_t_broadcast = z_lm1_t.unsqueeze(1).expand(batch_sizes, gates_size)

        s_recurrent_t = F.linear(h_l_tm1, w_rec, bias)
        if h_lp1_tm1 is not None :
            s_topdown_t = z_l_tm1_broadcast * F.linear(h_lp1_tm1, w_topdown)
        else :
            s_topdown_t = 0.0
        s_bottomup_t = z_lm1_t_broadcast * F.linear(h_lm1_t, w_bottomup)
        s_t = s_recurrent_t + s_topdown_t + s_bottomup_t
        f_t, i_t, o_t, g_t, z_tilde_t = [s_t[:, ii:ii+hidden_size] for ii in range(0, s_t.size(-1), hidden_size)]
        f_t = F.sigmoid(f_t)
        i_t = F.sigmoid(i_t)
        o_t = F.sigmoid(o_t)
        g_t = F.tanh(g_t)
        z_tilde_t = fn_z(z_tilde_t) # TODO : sigmoid or hard_sigmoid #
        z_t = st_bernoulli(z_tilde_t)
        ### CASE ####
        z_UPDATE = (1-z_l_tm1) * (z_lm1_t)
        z_COPY = (1-z_l_tm1) * (1-z_lm1_t)
        z_UPDATE = z_UPDATE.unsqueeze(1).expand(batch_sizes, hidden_size)
        z_COPY = z_COPY.unsqueeze(1).expand(batch_sizes, hidden_size)
        # UPDATE c_t (Eq. 2) #
        c_t = where(z_UPDATE, f_t * c_tm1 + i_t * g_t, where(z_COPY, c_tm1, i_t * g_t)) 
        # UPDATE h_t (Eq. 3) #
        h_t = where(z_COPY, h_l_tm1, o_t * F.tanh(c_t))
        return (h_t, c_t, z_t)
        pass
    pass


##### WRAPPER #####
class StatefulBaseCell(Module) :
    def __init__(self) :
        super(StatefulBaseCell, self).__init__()
        self._state = None
        pass

    def reset(self) :
        self._state = None

    @property
    def state(self) :
        return self._state

    @state.setter
    def state(self, value) :
        self._state = value

class StatefulLSTMCell(StatefulBaseCell) :
    def __init__(self, input_size, hidden_size, bias=True) :
        super(StatefulLSTMCell, self).__init__()
        self.rnn_cell = nn.LSTMCell(input_size, hidden_size, bias)
        pass

    @property
    def weight_hh(self) :
        return self.rnn_cell.weight_hh.t()

    @property
    def weight_ih(self) :
        return self.rnn_cell.weight_ih.t()

    @property
    def bias_hh(self) :
        return self.rnn_cell.bias_hh

    @property
    def bias_ih(self) :
        return self.rnn_cell.bias_ih

    def forward(self, input) :
        batch = input.size(0)
        if self.state is None :
            h0 = Variable(torchauto(self).FloatTensor(batch, self.rnn_cell.hidden_size).zero_())
            c0 = Variable(torchauto(self).FloatTensor(batch, self.rnn_cell.hidden_size).zero_())
            # h0, c0 #
            self.state = (h0, c0)

        self.state = self.rnn_cell(input, self.state)
        return self.state

class StatefulGRUCell(StatefulBaseCell) :
    def __init__(self, input_size, hidden_size, bias=True) :
        super(StatefulGRUCell, self).__init__()
        self.rnn_cell = nn.GRUCell(input_size, hidden_size, bias)
        pass

    @property
    def weight_hh(self) :
        return self.rnn_cell.weight_hh.t()

    @property
    def weight_ih(self) :
        return self.rnn_cell.weight_ih.t()

    @property
    def bias_hh(self) :
        return self.rnn_cell.bias_hh

    @property
    def bias_ih(self) :
        return self.rnn_cell.bias_ih

    def forward(self, input) :
        batch = input.size(0)
        if self.state is None :
            h0 = Variable(torchauto(self).FloatTensor(batch, self.rnn_cell.hidden_size).zero_())
            # h0, c0 #
            self.state = h0

        self.state = self.rnn_cell(input, self.state)
        return self.state
###################

###################
# StatefulLSTM (fast LSTM cuDNN)
class StatefulBase(Module) :
    def __init__(self) :
        super().__init__()
        self._state = None
        pass

    def reset(self) :
        self._state = None

    @property
    def state(self) :
        return self._state

    @state.setter
    def state(self, value) :
        self._state = value

class StatefulLSTM(StatefulBase) :
    def __init__(self, input_size, hidden_size, num_layers, bias=True, 
            batch_first=True, dropout=0, bidirectional=False) :
        super().__init__()
        assert batch_first == True, "please set batch_first == True"
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                num_layers=num_layers, bias=bias, batch_first=batch_first, 
                dropout=dropout, bidirectional=bidirectional)
        pass

    def forward(self, input) :
        if isinstance(input, Variable) :
            batch = input.shape[0]
        else :
            # case: input is PackedSequence
            batch = input.data.shape[0]

        if self.state is None :
            output, hidden = self.rnn(input)
        else :
            # swap batch axis
            _h0 = self.state[0].transpose(0, 1)
            _h1 = self.state[1].transpose(0, 1)
            output, hidden = self.rnn(input, (_h0, _h1))
        # swap batch axis
        _h0 = hidden[0].transpose(0, 1)
        _h1 = hidden[1].transpose(0, 1)
        self.state = (_h0, _h1)
        return output
