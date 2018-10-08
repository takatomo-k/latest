import torch
import numpy as np
from numbers import Number

def discount_reward(rewards, discount_rate=0.99) :
    discounted_rewards = []
    R = 0
    for tt in range(len(rewards)-1, -1, -1) :
        R = rewards[tt] + discount_rate*R
        discounted_rewards.append(R)
    discounted_rewards = list(reversed(discounted_rewards))
    return discounted_rewards
    pass

class BaselineFunction() :
    def __init__(self) :
        self.history = dict()
        pass
    pass

class TimeBaselineFunction(BaselineFunction) :
    def __init__(self, window=100, eps=1e-5):
        super().__init__()
        self.window = window
        self.eps = eps

    def prune(self, state) :
        self.history[state] = self.history[state][-self.window:]

    def append(self, state, value) :
        if state not in self.history :
            self.history[state] = []
        if isinstance(value, list) :
            self.history[state].extend(value)
        elif isinstance(value, Number) :
            self.history[state].append(value)
        else :
            raise ValueError("value type is not acceptable")
        pass

    def get_history(self, state) :
        return self.history.get(state, [])
        pass

    def mean_history(self, state) :
        records = self.get_history(state)[-self.window:]
        if len(records) == 0 :
            return 0
        else :
            return np.mean(records)
        pass

    def std_history(self, state) :
        records = self.get_history(state)[-self.window:]
        if len(records) == 0 :
            return 0
        else :
            return np.std(records)
        pass

