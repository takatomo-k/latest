import torch

from torch.autograd import Function

class Where(Function) :
    def forward(self, cond, x, y) :
        cond_f = cond if (isinstance(cond, torch.FloatTensor) 
                or isinstance(cond, torch.cuda.FloatTensor)) else cond.float()
        self.save_for_backward(cond_f)
        return cond_f * x + (1.0-cond_f) * y

    def backward(self, grad_output) :
        cond_f, = self.saved_tensors
        grad_cond = grad_x = grad_y = None
        grad_x = grad_output * cond_f
        grad_y = grad_output * (1.0 - cond_f)
        return grad_cond, grad_x, grad_y

    pass

def where(cond, x, y) :
    return Where()(cond, x, y)

def logsumexp(x):
    x_max, _ = x.max(dim=1,keepdim=True)
    x_max_expand = x_max.expand(x.size())
    res =  x_max + torch.log((x-x_max_expand).exp().sum(dim=1, keepdim=True))
    return res
