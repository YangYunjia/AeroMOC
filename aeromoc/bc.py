from .utils import *
import numpy as np
import math
from typing import Tuple, List, Dict, Callable
import matplotlib.pyplot as plt


class WallPoints():

    def __init__(self) -> None:
        self.xx = None
        self.yy = None
        self.dydx = None
        self.step = 1
        self.eps = 1e-2

    def __getitem__(self, index) -> Tuple[float, float, float]:
        return self.xx[index], self.yy[index], self.dydx[index]

    def __next__(self) -> Tuple[float, float, float]:
        if self.step >= len(self.xx):
            raise EndofwallError()
        else:
            self.step += 1
            return self[self.step - 1]

    def last(self) -> Tuple[float, float, float]:
        return self[self.step - 2]

    def add_section(self, xx: np.array, func: Callable, dfunc: Callable = None, relative_to_last: bool = True) -> None:
        
        _yy = np.array([func(xi) for xi in xx])
        
        if self.xx is not None and relative_to_last:
            _xx =  xx + self.xx[-1]
            _yy = _yy + self.yy[-1]
        else:
            _xx = xx
        
        if dfunc is None:
            # use centriod differencial to get dydx
            _dydx = np.array([(func(xi + EPS) - func(xi - EPS)) / (2.0 * EPS) for xi in xx])
        else:
            _dydx = np.array([dfunc(xi) for xi in xx])
        
        if self.xx is None:
            self.xx = _xx
            self.yy = _yy
            self.dydx = _dydx
        else:
            self.xx   = np.concatenate((self.xx,     _xx[min(1, len(xx)-1):]), axis=0)
            self.yy   = np.concatenate((self.yy,     _yy[min(1, len(xx)-1):]), axis=0)
            self.dydx = np.concatenate((self.dydx, _dydx[min(1, len(xx)-1):]), axis=0)
    
    def del_section(self, n: int) -> None:
        self.step = min(self.step, len(self.xx) - n)
        self.xx = self.xx[:-n]
        self.yy = self.yy[:-n]
        self.dydx = self.dydx[:-n]
        

    def plot(self):

        plt.figure(0)
        plt.plot(self.xx, self.yy, '-o', 'k')
        plt.show()