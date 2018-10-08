from ..nn.modules.bayes import ProbabilisticModule

BAYES_KLQP = 'klqp'
BAYES_LOGP = 'logp'
BAYES_LOGQ = 'logq'

def bayes_attr(module, attr_name) :
    total = 0
    for ii, mod in enumerate(module.modules()) :
        if isinstance(mod, ProbabilisticModule) :
            total += getattr(mod, attr_name)().sum()
        pass
    return total

def bayes_klqp(module) :
    return bayes_attr(module, BAYES_KLQP)

def bayes_logp(module) :
    return bayes_attr(module, BAYES_LOGP)

def bayes_logq(module) :
    return bayes_attr(module, BAYES_LOGQ)
