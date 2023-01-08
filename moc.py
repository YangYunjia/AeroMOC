'''
MOC

YangYunjia, 2023.1.3

'''
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Callable

GEOM = 0

BIG_NUMBER = 100000.

class ExtrapolateError(Exception):
    pass

class EndofwallError(Exception):
    pass

class Node():

    def __init__(self, *value) -> None:
        self.cors   = np.ones(2) * (-BIG_NUMBER)   # x, y
        self.vals   = np.array([-1.0, -1.0, -1.0, BIG_NUMBER])  # rho, p, vel, tta
        
        if len(value) in [2, 6]:
            self.cors = np.array(value[:2])
        if len(value) == 6:
            self.vals = np.array(value[2:6])

        self.g      = 1.4

    def __repr__(self) -> str:
        return '(%.2f, %.2f) (%.2f, %.2f %.2f, %.2f)' % tuple(list(self.cors) + list(self.vals))

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
        if self.y == 0.:
            return 0.
        return GEOM * math.sin(self.tta) / (self.y * self.ma * math.cos(self.tta + self.alp))

    @property
    def S_minus(self) -> float:
        '''
        calculate S minus(right) term in Mo's paper (3-24)
        '''
        if self.y == 0.:
            return 0.
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
    T_minus = _p1.Q * _p1.p - _p1.tta - _p1.S_minus * (_p4.x - _p1.x)
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
        T_minus = _p1.Q * _p1.p - _p1.tta - _p1.S_minus * (_p4.x - _p1.x)
        _p4.p = (T_minus + _p4.tta) / _p1.Q
    elif _p2 is not None:
        T_plus  = _p2.Q * _p2.p + _p2.tta - _p2.S_plus  * (_p4.x - _p2.x)
        _p4.p = (T_plus  - _p4.tta) / _p2.Q
    else:
        raise ValueError('When calculating boundary p4, at least one point should be set')

    _p4.vel = _p3.vel - (_p4.p - _p3.p) / (_p3.rho * _p3.vel)
    _p4.rho = _p3.rho + (_p4.p - _p3.p) / (_p3.a**2)

    return True

def calc_interior_point(p1: Node, p2: Node) -> Node:
    '''
    
    '''
    p4  = Node()
    _p1 = copy.deepcopy(p1)
    _p2 = copy.deepcopy(p2)

    # predict step

    _calc_p4_xy(_p1, _p2, p4)

    _calc_p4_vals(_p1, _p2, p4)
    _p1.vals = 0.5 * (_p1.vals + p4.vals)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _calc_p4_vals(_p1, _p2, p4)

    return p4

def calc_wall_point(xx: float, yy: float, dydx: float, p3: Node, p5: Node):
    
    p4  =  Node(xx, yy)
    p4.tta = math.atan(dydx)

    _p5 = copy.deepcopy(p5)
    _p3 = copy.deepcopy(p3)

    p2 = Node()

    ratio = 0.5
    ratio_old = 0.0
    # decide the point2 (origin of the LRC that ends at point4)
    while abs(ratio - ratio_old) > 1e-3:
        ratio_old = ratio
        p2.cors = _p5.cors + ratio * (_p3.cors - _p5.cors)
        p2.vals = _p5.vals + ratio * (_p3.vals - _p5.vals)
        ratio = _calc_p3_xy(_p3, _p5, p4, p2.lam_plus)
        if ratio > 1.0 or ratio < 0.0:
            raise ExtrapolateError()

    _p2 = copy.deepcopy(p2)
    _calc_boundary_p4_vals(_p3, p4, _p2=_p2)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _p3.vals = 0.5 * (_p3.vals + p4.vals)
    _calc_boundary_p4_vals(_p3, p4, _p2=_p2)

    return p4, p2
def calc_sym_point(p1: Node, p3: Node) -> Node:

    p4 = Node()
    _p1 = copy.deepcopy(p1)
    _p3 = copy.deepcopy(p3)

    p4.y = 0.0
    p4.x = p1.x + (p4.y - p1.y) / p1.lam_minus 

    p4.tta = 0.0
    
    _calc_boundary_p4_vals(_p3, p4, _p1=_p1)
    _p1.vals = 0.5 * (_p1.vals + p4.vals)
    _calc_boundary_p4_vals(_p3, p4, _p1=_p1)

    return p4


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
            _dydx = np.array([(func(xi + eps) - func(xi - eps)) / (2.0 * eps) for xi in xx])
        else:
            _dydx = np.array([dfunc(xi) for xi in xx])
        
        if self.xx is None:
            self.xx = _xx
            self.yy = _yy
            self.dydx = _dydx
        else:
            self.xx   = np.concatenate((self.xx,   _xx[1:]), axis=0)
            self.yy   = np.concatenate((self.yy,   _yy[1:]), axis=0)
            self.dydx = np.concatenate((self.dydx, _dydx[1:]), axis=0)
            
    def plot(self):

        plt.figure(0)
        plt.plot(self.xx, self.yy, '-o', 'k')
        plt.show()

        

if __name__ == '__main__':

    eps = 1e-2

    n = 8
    x0 = np.zeros(n)
    y0 = np.linspace(2., 0., n)
    print(len(y0))
    tta0 = np.zeros(n)
    p0 = np.ones(n) * 101325
    rho0 = np.ones(n) * 1.125
    vel0 = np.ones(n) * (1.4 * p0 / rho0)**0.5 * 1.1

    upperwall = WallPoints()
    upperwall.add_section(np.sin(np.linspace(0., math.pi / 12., 9)), lambda x: 3. - (1. - x**2)**0.5)
    upperwall.add_section(np.linspace(0, 5, 12), lambda x: math.tan(math.pi / 12.) * x)
    # upperwall.plot()


    init_line = [Node(x0[i], y0[i], rho0[i], p0[i], vel0[i], tta0[i]) for i in range(n)]
    sym_node = copy.deepcopy(init_line[-1])

    grid_points = []
    grid_points.append(init_line)

    step = 0
    max_step = 100
    flag_upper_wall = True

    while(len(init_line) > 0 and step < max_step):
        
        new_line = []
        # calculate interior points amount = (N_lastline - 1)
        for i in range(1, len(init_line)):
            new_node = calc_interior_point(init_line[i - 1], init_line[i])
            new_line.append(new_node)

            plt.plot([init_line[i - 1].x, new_node.x], [init_line[i - 1].y, new_node.y], '-', c='r')
            plt.plot([init_line[i].x, new_node.x], [init_line[i].y, new_node.y], '-', c='b')

        if step % 2 > 0:
            old_sym_node = copy.deepcopy(sym_node)
            sym_node = calc_sym_point(init_line[-1], old_sym_node)

            plt.plot([old_sym_node.x, sym_node.x], [old_sym_node.y, sym_node.y], '-', c='k')
            plt.plot([init_line[-1].x,  sym_node.x], [init_line[-1].y,  sym_node.y], '-', c='r')

            new_line += [sym_node]
        
        # upper boundary (wall)
        try:
            if flag_upper_wall:
                xw, yw, dydxw = next(upperwall)

            wall_node, intp_node = calc_wall_point(xw, yw, dydxw, init_line[0], new_line[0])

            plt.plot([upperwall.last()[0], wall_node.x], [upperwall.last()[1], wall_node.y], '-', c='k')
            plt.plot([intp_node.x,    wall_node.x], [intp_node.y,    wall_node.y], '-', c='b')

            new_line = [wall_node] + new_line
            flag_upper_wall = True

        except ExtrapolateError:
            flag_upper_wall = False
        except EndofwallError:
            pass

        init_line = copy.deepcopy(new_line)
        grid_points.append(copy.deepcopy(new_line))
        # print(step, len(init_line))
        step += 1

    # plt.xlim(0,1)
    plt.show()

    plt.plot(range(20), [grid_points[i][0].p for i in range(20)])
    # plt.plot(range(10), [grid_points[i][0].vel for i in range(10)])
    # plt.plot(range(10), [grid_points[i][0].a   for i in range(10)])
    # plt.plot(range(10), [grid_points[i][0].alp / 3.14 * 180 for i in range(10)])
    # plt.plot(range(10), [grid_points[i][0].tta / 3.14 * 180 for i in range(10)])
    plt.show()