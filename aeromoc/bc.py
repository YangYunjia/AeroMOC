from .utils import *
import numpy as np
import math
from typing import Tuple, List, Dict, Callable
import matplotlib.pyplot as plt


class WallPoints():

    '''
    `WallPoints` is a class for 2-D boundary conditions. Tt describes a line in 2-D plane by discrete 
    points. These points are saved in `np.array` called `self.xx` and `self.yy`, the steepness (dydx)
    is also stored.

    The wall is constructed by using `add_section`

    >>>     wall = WallPoints()
    >>>     xx = np.linspace(0, 5, 9)
    >>>     wall.add_section(xx, lambda x: -t * x)

    The `WallPoints` class is wrapped as an iterator. The coordinate value can be obtained by index and
    iteration:

    
    #### index
    >>>     x, y, dydx = wall[10]

    #### iteration
    When used by iteration, an utils.EndofwallError will be raised when reaches the end of wall.

    >>>     try:
    >>>         x, y, dydx = next(wall)
    >>>     except StopIteration:
    >>>         break

    or

    >>>     for x, y, dydx in wall:
    >>>         pass


    
    '''

    def __init__(self) -> None:
        self.xx = None
       
        self.dydx = None
        self.step = 1
        self.eps = 1e-2

    def __getitem__(self, index) -> Tuple[float, float, float]:
        return self.xx[index], self.yy[index], self.dydx[index]

    def __next__(self) -> Tuple[float, float, float]:
        if self.step >= len(self.xx):
            raise StopIteration()
        else:
            self.step += 1
            return self[self.step - 1]

    def last(self, idx=-1) -> Tuple[float, float, float]:
        '''
        Return the last wall point coordinate.
        
        Remark: after using `next()`, the index is one more then current point
        '''
        return self[self.step + idx - 1]

    def add_section(self, xx: np.array, yy: np.array = None, dydx: np.array = None,
                    func: Callable = None, 
                    dfunc: Callable = None, 
                    relative_to_last: bool = True) -> None:
        '''
        add new section to the wall

        ### para:

        - `xx`:     The x-direction coordinate of new section, the amount of points is decided by length of `xx`
        - `func`:   A function (`Callable`, either `lambda` or a function) to give y-dirction coordinate for each point in `xx`
        - `dfunc`:  A function to give gradient `dydx` for each point in `xx`
            - if the `dfunc` is None, then `dfunc` is obtained by finite difference (finite length is decided by `utils.EPS`)
        - `relative_to_last`:   (bool) whether the newly added points is relative to the last existing point.
        
        '''
        
        if yy is not None:
            _yy = yy
        else:
            _yy = np.array([func(xi) for xi in xx])
        
        if self.xx is not None and relative_to_last:
            _xx =  xx + self.xx[-1]
            _yy = _yy + self.yy[-1]
        else:
            _xx = xx
        
        if dfunc is not None:
            _dydx = np.array([dfunc(xi) for xi in xx])
        elif dydx is not None:
            _dydx = dydx
        else:
            # use centriod difference to get dydx
            if func is not None:
                _dydx = np.array([(func(xi + EPS) - func(xi - EPS)) / (2.0 * EPS) for xi in xx])
            else:
                _dydx = np.zeros_like(_xx)
                _dydx[1:-1] = (_yy[2:] - _yy[:-2]) / (_xx[2:] - _xx[:-2])
        
        if self.xx is None:
            self.xx = _xx
            self.yy = _yy
            self.dydx = _dydx
        else:
            self.xx   = np.concatenate((self.xx,     _xx[min(1, len(xx)-1):]), axis=0)
            self.yy   = np.concatenate((self.yy,     _yy[min(1, len(xx)-1):]), axis=0)
            self.dydx = np.concatenate((self.dydx, _dydx[min(1, len(xx)-1):]), axis=0)
        
        self.dydx = np.arctan(self.dydx)
    
    def del_section(self, n: int) -> None:
        '''
        delete n existing points of the wall

        ### para:
        - `n`:     number of deleting points
        '''

        self.step = min(self.step, len(self.xx) - n)
        self.xx = self.xx[:-n]
        self.yy = self.yy[:-n]
        self.dydx = self.dydx[:-n]
        
    def plot(self):

        # plt.figure(0)
        plt.plot(self.xx, self.yy, '-', c='k')
        # plt.show()
