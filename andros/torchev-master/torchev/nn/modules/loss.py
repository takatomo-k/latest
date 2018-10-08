import torch
from torch.nn import Module
import torch
from torch.nn import Module

class MaskedNLLLoss(Module) :
    def __init__(self, size_average=True, ignore_label=-1, label_smoothing=0) :
        super(MaskedNLLLoss, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        assert  0 <=label_smoothing < 1.0, "range label_smoothing should be [0, 1)" 
        self.label_smoothing = label_smoothing
    
    def forward(self, log_input, target) :
        # target -1 means ignored #
        log_input_target = log_input.gather(1, target.clamp(min=0).unsqueeze(1)).squeeze()
        target_excluding_ignored = (target != self.ignore_label).float()
        if self.size_average :
            count = target_excluding_ignored.sum()
        else :
            count = 1.0
        coeff = 1.0 / (count.clamp(min=1))
        if self.label_smoothing > 0 :
            smooth_pos = (1-self.label_smoothing)
            smooth_neg = self.label_smoothing / log_input.size(-1)
            loss = (log_input_target * target_excluding_ignored * smooth_pos) \
                    + ((log_input * smooth_neg).sum(dim=1) * target_excluding_ignored)
        else :
            loss = (log_input_target * target_excluding_ignored)
        loss = loss.sum() * -coeff
        return loss
    pass

def masked_nllloss(input, target, size_average=True, ignore_label=-1, label_smoothing=0) :
    return MaskedNLLLoss(size_average, ignore_label, label_smoothing)(input, target)

class MaskedCrossEntropyLoss(Module) :
    def __init__(self, size_average=True, ignore_label=-1, label_smoothing=0) :
        super(MaskedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing

    def forward(self, input, target) :
        return masked_nllloss(torch.nn.functional.log_softmax(input, dim=-1), target, 
                self.size_average, self.ignore_label, self.label_smoothing)
    pass

def masked_crossentropy(input, target, size_average=True, ignore_label=-1, label_smoothing=0) :
    return MaskedCrossEntropyLoss(size_average, ignore_label, label_smoothing)(input, target)
