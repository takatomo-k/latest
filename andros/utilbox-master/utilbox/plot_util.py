"""
plotutil -- DONT USE THIS, WILL BREAK YOUR CODE
"""

import time
import numpy as np
import os

__all__ = ['IterPlot', 'check_display']

def check_display() :
    if 'DISPLAY' not in os.environ :
        return False
    return True

import matplotlib
if not check_display() :
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


class IterPlot(object) :
    """
    IterPlot is a helper class for live time plotting multiple value
    Args :
        point_style (map): (k is the unit name, v is the plotting style)
        title (str): figure title
        xlim (int tuple): (left=x, right=y)
        ylim (int tuple): (bottom=x, top=y)

    Usage :
    >>> nx = np.sin(np.linspace(0, 6*np.pi, 100))
    >>> ny = np.cos(np.linspace(0, 3*np.pi, 100))
    >>> iplt = IterPlot({'cost_1':'g-', 'cost_2':'r-'}, xlim=(0,100))

    >>> for x,y in zip(nx, ny) :
    >>>     iplt.add_point({'cost_1':x, 'cost_2':y})
    >>>     time.sleep(0.1)


    """
    def __init__(self, point_style, title='Iter Plot', xlim=None, ylim=None) :
        self.point_style = point_style
        # setup plot #
        plt.ioff()
        self.fig = plt.figure()
        self.fig.suptitle(title, fontsize=14)
        self.ax = self.fig.add_subplot(111)
        if xlim != None :
            self.ax.set_xlim(*xlim)
        if ylim != None :
            self.ax.set_ylim(*ylim)
        self.point_obj = {}
        for k, v in list(point_style.items()) :
            points, = self.ax.plot([], [], v)
            self.point_obj[k] = points
            pass
        self.point_data = {}
        for k, v in list(point_style.items()) :
            self.point_data[k] = {'x':[], 'y':[]}
        self.ax.legend(list(self.point_obj.values()), list(self.point_obj.keys()))
        self.fig.show()
        pass

    def add_point(self, new_point) :
        """
        Args :
            new_point (map) : (k is the unit name, v is the value)
        """
        for k, v in list(new_point.items()) :
            if len(self.point_data[k]['x']) == 0 :
                self.point_data[k]['x'] = [0]
            else :
                self.point_data[k]['x'].append(self.point_data[k]['x'][-1]+1)
            self.point_data[k]['y'].append(v)
            self.point_obj[k].set_xdata(self.point_data[k]['x'])
            self.point_obj[k].set_ydata(self.point_data[k]['y'])
            pass
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        pass

    def refresh(self) :
        self.fig.canvas.flush_events()
        pass

    def save(self, filename) :
        if filename == None :
            filename = self.title
        self.fig.savefig('%s'%(filename))
        pass

if __name__ == '__main__':
    nx = np.sin(np.linspace(0, 6*np.pi, 100))
    ny = np.cos(np.linspace(0, 3*np.pi, 100))
    # test #
    iplt = IterPlot({'cost_1':'g-', 'cost_2':'r-'}, xlim=(0,100))

    for x,y in zip(nx, ny) :
        iplt.add_point({'cost_1':x, 'cost_2':y})
        iplt.save('/tmp/test_plotutil.png')
        time.sleep(0.1)
    pass
