import pytest
import numpy as np
import torch
from torch.autograd import Variable
from torchev.nn.modules.loss_elementwise import ElementwiseNLLLoss, ElementwiseCrossEntropy
np.random.seed(123)
BATCH = 50
NCLASS = 5

class TestElementwiseNLLLoss(object) :
    def test_fwbw(self) :
        input = Variable(torch.randn(BATCH, NCLASS))
        target = Variable(torch.from_numpy(np.random.randint(0, NCLASS, size=(BATCH,))))
        cost_mnll = ElementwiseNLLLoss()(torch.nn.functional.log_softmax(input, dim=-1), target).mean()
        cost_nll = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(input, dim=-1), target)
        assert np.allclose(cost_mnll.data.numpy(), cost_nll.data.numpy())
        pass
    def test_fwbw_plus_weight(self) :
        input = Variable(torch.randn(BATCH, NCLASS))
        target = Variable(torch.from_numpy(np.random.randint(0, NCLASS, size=(BATCH,))))
        _weight = torch.FloatTensor(NCLASS).zero_()+1.0
        # _weight[3] = 0
        weight = Variable(_weight)
        cost_mnll = ElementwiseNLLLoss(weight=weight)(torch.nn.functional.log_softmax(input, dim=-1), target).mean()
        cost_nll = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(input, dim=-1), target, weight=weight.data)
        assert np.allclose(cost_mnll.data.numpy(), cost_nll.data.numpy())
        pass
    def test_fwbw_ignore(self) :
        input = Variable(torch.randn(BATCH+10, NCLASS))
        target = Variable(torch.from_numpy(np.random.randint(0, NCLASS, size=(BATCH+10,))))
        mask = Variable(torch.ones(BATCH+10))
        mask[-10:] = 0
        cost_mnll = ElementwiseNLLLoss()(torch.nn.functional.log_softmax(input, dim=-1), target) * mask
        cost_mnll = cost_mnll.sum().div(mask.sum())
        cost_nll = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(input[0:BATCH], dim=-1), target[0:BATCH])
        assert np.allclose(cost_mnll.data.numpy(), cost_nll.data.numpy())
        pass
    def test_fwbw_ignore_plus_weight(self) :
        input = Variable(torch.randn(BATCH+10, NCLASS))
        target = Variable(torch.from_numpy(np.random.randint(0, NCLASS, size=(BATCH+10,))))
        mask = Variable(torch.ones(BATCH+10))
        mask[-10:] = 0
        _weight = torch.FloatTensor(NCLASS).zero_()+1.0
        # _weight[3] = 0
        weight = Variable(_weight)
        cost_mnll = ElementwiseNLLLoss(weight=weight)(torch.nn.functional.log_softmax(input, dim=-1), target) * mask
        cost_mnll = cost_mnll.sum().div(mask.sum())
        cost_nll = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(input[0:BATCH], dim=-1), target[0:BATCH], weight=weight.data)
        assert np.allclose(cost_mnll.data.numpy(), cost_nll.data.numpy())
        pass
    def test_fwbw_ignore_lblsmooth(self) :
        LABEL_SMOOTHING = 0.1
        input = Variable(torch.randn(BATCH+10, NCLASS), requires_grad=True)
        target = Variable(torch.from_numpy(np.random.randint(0, NCLASS, size=(BATCH+10,))))
        mask = Variable(torch.ones(BATCH+10))
        mask[-10:] = 0
        log_input = torch.nn.functional.log_softmax(input, dim=-1)
        cost_mnll = ElementwiseNLLLoss(label_smoothing=LABEL_SMOOTHING)(log_input, target) * mask
        cost_mnll = cost_mnll.sum().div(mask.sum())
       
        t_input = Variable(input[0:BATCH].data.clone(), requires_grad=True)
        t_log_input = torch.nn.functional.log_softmax(t_input, dim=-1)
        t_target_onehot = torch.FloatTensor(BATCH, NCLASS).zero_()
        t_target_onehot.scatter_(1, target.data[0:BATCH].unsqueeze(1), 1-LABEL_SMOOTHING)
        t_target_onehot += LABEL_SMOOTHING / NCLASS
        t_target_onehot = Variable(t_target_onehot)
        cost_nll =  -(t_log_input * t_target_onehot).sum() / BATCH
        assert np.allclose(cost_mnll.data.numpy(), cost_nll.data.numpy())
        cost_mnll.backward()
        cost_nll.backward()
        assert np.allclose(input.grad.data[0:BATCH].numpy(), t_input.grad.data.numpy())
        pass
    pass
