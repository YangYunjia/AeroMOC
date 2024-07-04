from .utils import *
import numpy as np
import math
from typing import Tuple, List, Dict, Callable
import matplotlib.pyplot as plt

def finite_diff(_xx: np.ndarray, _yy: np.ndarray):
    _dydx = np.zeros_like(_xx)
    _dydx[1:-1] = (_yy[2:] - _yy[:-2]) / (_xx[2:] - _xx[:-2])
    _dydx[0]    = (_yy[1] - _yy[0])    / (_xx[1] - _xx[0])
    _dydx[-1]   = (_yy[-1] - _yy[-2])  / (_xx[-1] - _xx[-2])
    
    return _dydx

class BoundPoints():

    '''
    `BoundPoints` is a class for 2-D boundary conditions. Tt describes a line in 2-D plane by discrete 
    points. These points are saved in `np.array` called `self.xx` and `self.yy`, the steepness (dydx)
    is also stored.

    The boundary is constructed by using `add_section`

    >>>     wall = BoundPoints()
    >>>     xx = np.linspace(0, 5, 9)
    >>>     wall.add_section(xx, lambda x: -t * x)

    The `BoundPoints` class is wrapped as an iterator. The coordinate value can be obtained by index and
    iteration:

    
    #### index
    >>>     x, y, dydx = wall[10]

    #### iteration
    When used by iteration, an StopIteration will be raised when reaches the end of wall.

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
                _dydx = finite_diff(_xx, _yy)
        
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

'''
Boundary-layer correction (BLC)
'''

'''
class BLC():

    def __init__(self, method) -> None:
        self.method = method

def blc(xx: np.array, x0: float, method: str, **kwargs):

    if method == 'linear':
        return blc_linear_estimate(xx, x0, **kwargs)
    elif method == 'edenfield':
        return blc_edenfield(xx, x0, **kwargs)
'''

def blc_linear_estimate(xx: np.array, x0: float, Me: float):
    
    '''
    Use linear relation to estimate thickness
    xx:     x-coordinates
    x0:     the origin of the boundary layer
    Me:     Mach number at the exit of the nozzle
    Ref:    刘政崇. 高低速风洞气动与结构设计[M]. 国防工业出版社, 2003
    '''

    a0 = -7.166650
    a1 = 6.694431
    a2 = -2.209718
    a3 = 0.3385411
    a4 = -2.3611065e-2
    a5 = 6.0763751e-4

    beta = (a0 + a1 * Me + a2 * Me**2 + a3 * Me**3 + a4 * Me**4 + a5 * Me**5)

    return (xx - x0) * np.tan(beta / 180 * np.pi)

def blc_edenfield(xx: np.array, x0: float, pp: np.array, ma: np.array, tt: np.array, t0: float, tw: float = None):
    '''
    Apply the boundary layer correction with Edenfield's method
    xx, pp, ma, tt:     the x-coordinate, pressure, mach number, and temperature obtained with MOC, 
                        will be seen as the margin value of the boundary layer
    x0:     the origin of the boundary layer
    t0:     inlet total temperature
    tw:     the wall temperature, `None` for adiabatic wall
    
    Ref: Edenfield E E. Contoured nozzle design and evaluation for hotshot wind tunnels. AIAA68-0369, 1968.04.
    
    Remark: Due to the low temperature of the wind tunnel, Saterland's formula is not fully applicable, so the 
            formula given by Brebach Todos can be considered for calculation.
            Ref. Brebach W J, Thodos G. Viscosity-reduced state correlation for diatomic gases. Ind. Eng. Chem, Vol.50, NO.7, 1958.07.
    '''

    r = GAS_PR**(1./3.)           # recover coefficents

    t_ad = tt + r * (t0 - tt) # adiatic recover temperature

    if tw is None:  # adiatic wall
        tw = t_ad
    
    t_ref   = 0.5 * (tw + tt) + 0.22 * (t_ad - tt)   # reference temerature
    mu_ref  = sutherland(t_ref);    # viscosity
    rho_ref = pp / (t_ref * GAS_R);   # density corresponding to the t_ref
    Ue      = ma * (GAS_GAMMA * GAS_R * tt)**0.5  # velocity at the margin of the boundary layer
    Re_ref  = rho_ref * Ue * (xx - x0) / mu_ref # reference Reynold number

    return 0.42 * (xx - x0) * Re_ref**-0.2775

