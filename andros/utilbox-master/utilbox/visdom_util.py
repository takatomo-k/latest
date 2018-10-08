from visdom import Visdom
import numpy as np 

class VisLine :
    def __init__(self, opts={}) :
        self.prev_x = 0
        self.win = None
        self.viz = Visdom()
        self.opts = opts
        pass
    def append(self, y, x=None) :
        if x is None :
            x = np.arange(self.prev_x, self.prev_x+len(y))
            self.prev_x += len(y)
        if self.win is None :
            self.win = self.viz.line(y, x, opts=self.opts)
        else :
            self.viz.line(y, x, win=self.win, update='append')
        pass
