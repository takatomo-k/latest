import torch

from torch.autograd import Function, Variable
from torch.nn import functional as F
from torch.nn import Module
from ...utils.helper import torchauto, tensorauto

def hard_sigmoid(input, slope=0.2, shift=0.5) :
    output = (input * slope) + shift
    output = torch.clamp(output, min=0, max=1)
    return output

class STBernoulli(Function) :
    def __init__(self, stochastic=True) :
        self.stochastic = stochastic
        pass

    def forward(self, input) :
        # assert 0 <= input <= 1 # 
        if self.stochastic :
            output = input.bernoulli()
        else :
            output = (input > 0.5).float()
        return output

    def backward(self, grad_output) :
        grad_input = grad_output.clone()
        return grad_input
    pass

def st_bernoulli(input, stochastic=True) :
    return STBernoulli(stochastic)(input)

class STMultinomial(Function) :
    def __init__(self, stochastic=True) :
        self.stochastic = stochastic
        pass

    def forward(self, input) :
        # TODO : use sparse tensor for efficient memory usage #
        # retain original shape after sampling into one-hot encoding #
        if self.stochastic :
            output = input.multinomial(1, replacement=True)
        else :
            output = input.max(1)[1]
        output = torchauto(input).FloatTensor(input.size()).zero_().scatter_(1, output, 1.0)
        return output

    def backward(self, grad_output) :
        grad_input = grad_output.clone()
        return grad_input

def st_multinomial(input, stochastic=True) :
    return STMultinomial(stochastic)(input)

# TODO reinforce bernoulli 
# TODO reinforce multinomial

# Gumbel Softmax #
def _sample_gumbel(shape, eps=1e-15) :
    noise = torch.rand(shape)
    return -torch.log(-torch.log(noise + eps) + eps)

def gumbel_softmax_sample(logits, temperature) :
    noise = Variable(tensorauto(logits.data, _sample_gumbel(logits.size())))
    y = (logits + noise) / temperature
    return F.softmax(y)

def gumbel_softmax(logits, temperature, hard=False) :
    y = gumbel_softmax_sample(logits, temperature)
    if hard :
        _, max_idx = y.data.max(dim=1)
        y_hard = torchauto(logits.data).FloatTensor(y.size()).zero_().scatter_(1, max_idx, 1.0)
        y_hard = Variable(y_hard)
        y = (y_hard - y).detach() + y
    return y


# dummy activation #
class Identity(Module) :
    def __init__(self) :
        super().__init__()
        pass
    
    def forward(self, input) :
        return input

    def __repr__(self) :
        return self.__class__.__name__+'()'
