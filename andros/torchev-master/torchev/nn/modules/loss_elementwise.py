import torch
from torch.nn import Module

class ElementwiseNLLLoss(Module) :
    def __init__(self, label_smoothing=0, weight=None) :
        super(ElementwiseNLLLoss, self).__init__()
        assert  0 <=label_smoothing < 1.0, "range label_smoothing should be [0, 1)" 
        self.label_smoothing = label_smoothing
        self.weight = weight
    
    def forward(self, log_input, target) :
        # target -1 means ignored #
        log_input_target = log_input.gather(1, target.unsqueeze(1)).squeeze()
        if self.label_smoothing > 0 :
            smooth_pos = (1-self.label_smoothing)
            smooth_neg = self.label_smoothing / log_input.size(-1)
            loss = (log_input_target * smooth_pos) \
                    + ((log_input * smooth_neg).sum(dim=1))
        else :
            loss = log_input_target

        if self.weight is not None :
            loss = loss * self.weight.index_select(0, target)

        loss = -loss # negative LL # 
        return loss
    pass

def elementwise_nllloss(input, target, label_smoothing=0, weight=None) :
    return ElementwiseNLLLoss(label_smoothing=label_smoothing, weight=weight)(input, target)

class ElementwiseCrossEntropy(Module) :
    def __init__(self, label_smoothing=0, weight=None) :
        super(ElementwiseCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight

    def forward(self, input, target) :
        return elementwise_nllloss(torch.nn.functional.log_softmax(input, dim=-1), target, 
                label_smoothing=self.label_smoothing, weight=self.weight)
    pass

def elementwise_crossentropy(input, target, label_smoothing=0, weight=None) :
    return ElementwiseCrossEntropy(label_smoothing, weight)(input, target)

class ElementwiseBCE(Module) :
    def __init__(self) :
        super().__init__()
        pass

    def forward(self, input, target) :
        return -(target*torch.log(input) + (1-target)*torch.log(1-input))
    pass

def elementwise_bce(input, target) :
    return ElementwiseBCE()(input, target)

class ElementwiseBCEWithLogits(Module) :
    def __init__(self) :
        super().__init__()
        pass

    def forward(self, input, target) :
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        return loss

def elementwise_bce_with_logits(input, target) :
    return ElementwiseBCEWithLogits()(input, target)
