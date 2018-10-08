import torch
from torch.autograd import Variable

def is_cuda_module(module) :
    return next(module.parameters()).is_cuda

def _auto_detect_cuda(module) :
    if isinstance(module, torch.nn.Module) :
        return is_cuda_module(module) 
    if isinstance(module, bool) :
        return module
    if isinstance(module, int) :
        return module >= 0
    if isinstance(module, torch.autograd.Variable) :
        return module.data.is_cuda
    if isinstance(module, torch.tensor._TensorBase) :
        return module.is_cuda
    raise NotImplementedError()


def torchauto(module) :
    return torch.cuda if _auto_detect_cuda(module) else torch

def tensorauto(module, tensor) :
    return tensor.cuda() if _auto_detect_cuda(module) else tensor.cpu()

def apply_fn_variables(variables, fn) :
    if variables is None :
        _res = None
    elif isinstance(variables, Variable) :
        _res = fn(variables)
    elif isinstance(variables, (list, tuple)) :
        _res = type(variables)(apply_fn_variables(x, fn) for x in variables)
    elif isinstance(variables, dict) :
        _res = dict((x, apply_fn_variables(y, fn)) for x, y in variables.items())
    else :
        raise ValueError
    return _res

def detach_variables(variables) :
    return apply_fn_variables(variables, (lambda x : x.detach()))
def clone_variables(variables) :
    return apply_fn_variables(variables, (lambda x : Variable(x.data)))

def vars_index_select(variables, dim, index, clone=False) :
    """
    function for index_select multiple variables (e.g decoder states)
    """
    if variables is None :
        _res = None
    elif isinstance(variables, Variable) :
        _res = variables.index_select(dim=dim, index=index)
        if clone :
            _res = variables.clone()
    elif isinstance(variables, (list, tuple)) :
        _res = type(variables)(vars_index_select(x, dim, index, clone) for x in variables)
    elif isinstance(variables, dict) :
        _res = dict((x, vars_index_select(y, dim, index, clone)) for x, y in variables.items())
    else :
        raise ValueError
    return _res


