class ExponentialDecay(object) :
    def __init__(self, init_value=0, global_step=0, decay_step=1, decay_rate=0.95, min_value=0) :
        self.init_value = init_value
        self.global_step = global_step
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.min_value = min_value
        pass
    
    @property
    def value(self) :
        return max(self.init_value * self.decay_rate ** (self.global_step / self.decay_step), self.min_value)
    
    def step(self, step=None) :
        if step is None :
            self.global_step += 1
        else :
            self.global_step = step

class LinearDecay(object) :
    def __init__(self, init_value=1, end_value=0, global_step=0, decay_step=1, decay_rate=0.05) :
        self.init_value = init_value
        self.end_value = end_value
        assert init_value > end_value, "initial value must be larger than the end value"
        self.global_step = global_step
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        pass
    
    @property
    def value(self) :
        return max(self.init_value - self.decay_rate * (self.global_step / self.decay_step), self.end_value)
    
    def step(self, step=None) :
        if step is None :
            self.global_step += 1
        else :
            self.global_step = step
