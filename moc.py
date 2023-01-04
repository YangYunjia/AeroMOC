'''
MOC

YangYunjia, 2023.1.3

'''
import math
import copy
import numpy as np


GEOM = 0
BIG_NUMBER = 100000.


class Node():

    def __init__(self) -> None:
        self.cors   = np.ones(4) * (-BIG_NUMBER)   # x, y
        self.vals   = np.array([-1.0, -1.0, -1.0, BIG_NUMBER])  # rho, p, vel, tta
        self.g      = 1.4

    @property
    def x(self) -> float:
        if self.cors[0] > -BIG_NUMBER: 
            return self.cors[0]
        else:
            raise ValueError('Value for x is not correct')
    
    @x.setter
    def x(self, value):
        self.cors[0] = value

    @property
    def y(self) -> float:
        if self.cors[1] > -BIG_NUMBER: 
            return self.cors[1]
        else:
            raise ValueError('Value for y is not correct')
    
    @y.setter
    def y(self, value):
        self.cors[1] = value

    @property
    def rho(self) -> float:
        if self.vals[0] > 1e-3: 
            return self.vals[0]
        else:
            raise ValueError('Value for rho = %.3f is not correct' % self.vals[0])
    
    @rho.setter
    def rho(self, value):
        self.vals[0] = value

    @property
    def p(self) -> float:
        if self.vals[1] > 1e-3: 
            return self.vals[1]
        else:
            raise ValueError('Value for p = %.3f is not correct' % self.vals[1])
    
    @p.setter
    def p(self, value):
        self.vals[1] = value

    @property
    def vel(self) -> float:
        if self.vals[2] > 1e-3: 
            return self.vals[2]
        else:
            raise ValueError('Value for velocity = %.3f is not correct' % self.vals[2])
    
    @vel.setter
    def vel(self, value):
        self.vals[2] = value

    @property
    def tta(self) -> float:
        if self.vals[3] < math.pi and self.vals[3] > -math.pi: 
            return self.vals[3]
        else:
            raise ValueError('Value for theta = %.3f is not correct' % self.vals[3])
    
    @tta.setter
    def tta(self, value):
        self.vals[3] = value

    @property
    def alp(self) -> float:
        '''
        calculate mach angle alpha
        '''
        return math.asin(1. / self.ma)
    
    @property
    def lam_plus(self) -> float:
        '''
        calculate the characteristic angle for Left Runing Characteristic from this node(positive)
        '''
        return math.tan(self.tta + self.alp)

    @property
    def lam_minus(self) -> float:
        '''
        calculate the characteristic angle for Left Runing Characteristic from this node(positive)
        '''
        return math.tan(self.tta - self.alp)

    @property
    def Q(self) -> float:
        '''
        calculate Q term in Mo's paper (3-24)
        '''
        return (self.ma**2 - 1)**0.5 / (self.rho * self.vel**2)

    @property
    def S_plus(self) -> float:
        '''
        calculate S plus(left) term in Mo's paper (3-24)
        '''
        return GEOM * math.sin(self.tta) / (self.y * self.ma * math.cos(self.tta + self.alp))

    @property
    def S_minus(self) -> float:
        '''
        calculate S minus(right) term in Mo's paper (3-24)
        '''
        return GEOM * math.sin(self.tta) / (self.y * self.ma * math.cos(self.tta - self.alp))

    @property
    def a(self) -> float:
        '''
        the acoustic speed
        '''
        return (self.g * self.p / self.rho)**0.5

    @property
    def ma(self) -> float:
        return self.vel / self.a
        
def _calc_p3_xy(_p1: Node, _p2: Node, _p4: Node, tta0: float) -> float:
    return (_p4.y - _p2.y - tta0 * (_p4.x - _p2.x)) / (_p1.y - _p2.y - tta0 * (_p1.x - _p2.x))

def _calc_p4_xy(_p1: Node, _p2: Node, _p4: Node) -> bool:
    _p4.x = ((_p1.y - _p2.y) - (_p1.lam_minus * _p1.x - _p2.lam_plus * _p2.x)) / (_p2.lam_plus - _p1.lam_minus)
    _p4.y = _p2.y + _p2.lam_plus * (_p4.x - _p2.x)
    return True

def _calc_p4_vals(_p1: Node, _p2: Node, _p4: Node) -> bool:
    _p3 = Node()

    T_plus  = _p2.Q * _p2.p + _p2.tta - _p2.S_plus  * (_p4.x - _p2.x)
    T_minus = _p1.Q * _p1.p + _p1.tta - _p1.S_minus * (_p4.x - _p1.x)
    _p4.p = (T_plus + T_minus) / (_p1.Q + _p2.Q)
    _p4.tta = T_plus - _p2.Q * _p4.p

    # the ratio of point 3 on the line p1-p2 from p2
    ratio = _calc_p3_xy(_p1, _p2, _p4, _p4.tta)

    _p3.vals = _p2.vals + ratio * (_p1.vals - _p2.vals)

    _p4.vel = _p3.vel - (_p4.p - _p3.p) / (_p3.rho * _p3.vel)
    _p4.rho = _p3.rho + (_p4.p - _p3.p) / (_p3.a**2)

    return True

def _calc_boundary_p4_vals(_p3: Node, _p4: Node, _p1: Node = None, _p2: Node = None) -> bool:
    
    if _p1 is not None:
        T_minus = _p1.Q * _p1.p + _p1.tta - _p1.S_minus * (_p4.x - _p1.x)
        _p4.p = (T_minus - _p4.tta) / _p1.Q
    elif _p2 is not None:
        T_plus  = _p2.Q * _p2.p + _p2.tta - _p2.S_plus  * (_p4.x - _p2.x)
        _p4.p = (T_plus  - _p4.tta) / _p2.Q
    else:
        raise ValueError('When calculating boundary p4, at least one point should be set')

    _p4.vel = _p3.vel - (_p4.p - _p3.p) / (_p3.rho * _p3.vel)
    _p4.rho = _p3.rho + (_p4.p - _p3.p) / (_p3.a**2)

    return True

def calc_interior_point(p1: Node, p2: Node, p4: Node) -> bool:
    '''
    
    '''
    _p1 = copy.deepcopy(p1)
    _p2 = copy.deepcopy(p2)

    # predict step

    _calc_p4_xy(_p1, _p2, p4)

    _calc_p4_vals(_p1, _p2, p4)
    _p1.vals = 0.5 * (_p1.vals + p4.vals)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _calc_p4_vals(_p1, _p2, p4)

    return True

def calc_wall_point(p3: Node, p4: Node, p5: Node) -> bool:
    
    _p5 = copy.deepcopy(p5)
    _p3 = copy.deepcopy(p3)

    _p2 = Node()

    ratio = 0.5
    ratio_old = 0.0
    # decide the point2 (origin of the LRC that ends at point4)
    while abs(ratio - ratio_old) > 1e-3:
        ratio_old = ratio
        _p2.vals = _p5.vals + ratio * (_p3.vals - _p5.vals)
        ratio = _calc_p3_xy(_p3, _p5, p4, _p2.lam_plus)

    _calc_boundary_p4_vals(_p3, p4, _p2=_p2)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _calc_boundary_p4_vals(_p3, p4, _p2=_p2)

    return True

def calc_sym_point(p1: Node, p3: Node, p4: Node) -> bool:

    _p1 = copy.deepcopy(p1)
    _p3 = copy.deepcopy(p3)

    p4.y = 0.0
    p4.x = p1.x + (p4.y - p1.y) / p1.lam_minus 

    p4.tta = 0.0
    
    _calc_boundary_p4_vals(_p3, p4, _p1=_p1)
    _p1.vals = 0.5 * (_p1.vals + p4.vals)
    _calc_boundary_p4_vals(_p3, p4, _p1=_p1)

    return True


