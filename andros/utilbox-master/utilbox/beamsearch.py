__author__ = "@arkmagus"

import numpy as np
import heapq

# TEST CASE #
trans_prob = np.array(
        [[0.3, 0.1, 0.6],
         [0.6, 0.5, 0.3],
         [0.1, 0.4, 0.1]]
        ) # p(state[t] | state[t-1]) 

NUM_CLASS = int(trans_prob.shape[0])
WIDTH = 5

def f_pred(prev_states) :
    # TODO use prev_states
    new_pred = np.random.rand(NUM_CLASS)
    new_pred /= new_pred.sum()
    return new_pred
    pass

class BaseSampler(object) :
    def __init__(self) :
        pass

    def sample(self, *args) :
        raise NotImplementedError("implement sampler which return BaseState object")

    pass

class BaseState(object) :
    def __init__(self) :
        self._score = 0.0
    
    @property
    def score(self) :
        return self._score
    
    @score.setter
    def score(self, value) :
        self._score = 0.0

    def __lt__(self, other) : 
        # reversed case for heapq
        return self.score > other.score
    def __cmp__(self, other) :
        return cmp(other.score, self.score)
    pass

# EXAMPLE CLASS #
class CustomState(BaseState) :
    def __init__(self, label, log_prob) :
        self.label = label
        self.log_prob = log_prob
        pass
    
    @property
    def score(self) :
        return self.log_prob
    
    @score.setter 
    def score(self, value):
        self.log_prob = value

    def __repr__(self) :
        return ("CustomState[label:%s, ln(prob):%f]"%(','.join(map(str, self.label)), self.log_prob))
        pass

# EXAMPLE CLASS #
class CustomSampler(BaseSampler) :
    def __init__(self, trans_prob) :
        self.trans_prob = trans_prob
        pass

    def sample(self, prev_state) :
        new_states = []
        for ii in range(NUM_CLASS) :
            new_label = prev_state.label + [ii]
            new_log_prob = prev_state.log_prob + np.log(trans_prob[ii][prev_state.label[-1]])
            new_states.append(CustomState(new_label, new_log_prob))
            pass
        pass
        return new_states
    pass

class BeamSearch(object) :
    def __init__(self, states, capacity=5, check_goal=None) :
        self._states = list(states)
        heapq.heapify(self._states)
        
        self._final_states = []
        self.capacity = capacity
        self.check_goal = check_goal
        pass
    
    @property
    def states(self) :
        return self._states
    
    def clear_states(self) :
        self._states = []
        pass

    def set_states(self) :
        self._states = list(states)
        heapq.heapify(self._states)
        pass

    @property
    def final_states(self) :
        return sorted(self._final_states)

    def append(self, state) :
        if not isinstance(state, BaseState) :
            raise ValueError("state is not BaseState object")
        heapq.heappush(self._states, state)
        pass

    def extend(self, states) :
        if not isinstance(states, list) :
            raise ValueError("states is not a list object")
        for state in states :
            self.append(state)
            pass

    def prune(self) :
        # remain top-K #
        _tmp = []
        for _ in range(self.capacity) :
            _tmp.append(heapq.heappop(self._states))
        self._states = [] 

        if self.check_goal is not None :    
            for _state in _tmp :
                if self.check_goal(_state) :
                    self._final_states.append(_state)
                    self.capacity -= 1
                else :
                    self._states.append(_state)
            pass

        heapq.heapify(self._states)
        pass

    def pop(self) :
        return heapq.heappop(self._states)
        pass

    pass


if __name__ == '__main__' :
    bs = BeamSearch([], 5) 
    bs.append(CustomState([0], np.log(0.3)))
    bs.append(CustomState([1], np.log(0.5)))
    bs.append(CustomState([2], np.log(0.2)))
    sampler = CustomSampler(trans_prob)
    counter = 0
    while len(bs.states) != 0 :
        state = bs.pop()
        new_states = sampler.sample(state)
        bs.extend(new_states)
        bs.prune()
        if counter == 100 :
            break
        counter += 1
        pass
           
    for x in bs.states :
       print(x)
    pass
