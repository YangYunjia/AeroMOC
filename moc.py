'''
MOC

YangYunjia, 2023.1.3

'''
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Callable

GEOM = 0
GAS_R = 1.4
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

    def set_by_total(self, tta: float, ma: float, pt: float, tt: float):
        p, t, rho = calc_isentropicPTRHO(g=self.g, ma=ma, pTotal=pt, tTotal=tt)
        self.rho = rho
        self.p   = p
        self.vel = ma * (self.g * GAS_R * t)**0.5
        self.tta = tta

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

def calc_isentropicPTRHO(g: float, ma: float, pTotal: float, tTotal: float):
    ratio = 1 + (g - 1) / 2. * ma**2
    p = pTotal / ratio**(g / (g - 1))
    t = tTotal / ratio
    rho = p / (GAS_R * t)
    return p, t, rho

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

def calc_initial_throat_line(n: int, yw0: float, yc0: float = 0.0,
             isTotal: bool = True, rUp: float = 0.0, pTotal: float = 0.0, tTotal: float = 0.0, mT: float = 0.0) -> List[Node]:
    '''
    This function is interpreted from code of CalcInitialThroatLine <- MOC_GidCalc_BDE <- the MOC programma of NASA
    '''

    _init_line: List[Node] = []

    for i in range(n + 1):
        # Based on the way it is done in RAO, the y[i] is based on a sinusoidal
        # and x[i] is assumed to be on the RRC from the throat wall
        
        # input yy should be non-dimensional
        _yy = math.sin(math.pi * (n - i) / n / 2.)**1.5

        if i > 0:
            dydx = _init_line[-1].lam_minus   # this calculates a new X first assumed it falls on the RRC

        _mach = 2.0
        point = Node()
        if isTotal:
            if i > 0:
                while _mach > 1.5:
                    _xx = _xx_old + (_yy - _yy_old) / dydx
                    _theta, _mach = KLThroat(_xx, _yy, point.g, rUp)
                    # increase dydx so that slope is half of what is was, only when mach number exceed 1.5
                    # this is to prevent the initial line is too steep and lead to solution failure
                    dydx *= 2.0
            else:
                _xx = 0.0
                _theta, _mach = KLThroat(_xx, _yy, point.g, rUp)

        else:
            if mT < 1.0:
                raise ValueError("Calculated Throat Mach number < 1.0\nCalcInitialThroatLine")
            _theta = 0.0
            _mach  = mT
            if i > 0:
                _xx = _xx_old.x + (_yy - _yy_old) / dydx
            else:
                _xx = 0.0
        
        point.x = (yw0 - yc0) * _xx
        point.y = (yw0 - yc0) * _yy + yc0
        _xx_old = _xx
        _yy_old = _yy
        # print(i, _theta, _mach)
        point.set_by_total(tta=_theta, ma=_mach, pt=pTotal, tt=tTotal)

        _init_line.append(point)

    return _init_line
                
def KLThroat(x: float, y: float, G: float, RS: float) -> None:
    '''
    adapted from `int MOC_GridCalc::KLThroat(int i, int geom, double RS)` <- MOC_GidCalc_BDE <- the MOC programma of NASA
    
    This is taken from 'Transonic Flow in Small Throat Radius Curvature Nozzles',
    by Kliegel and Levine. In this they take the HALL method and modify the Axi
    calculation for a toroid coordinate system.  The 2D version is the one developed by HALL.

    param:
    ---
    `y` is the radial distance of a point on the starting plane

    '''

    u = np.zeros(4)
    v = np.zeros(4)

    if (GEOM == 0):
        #  Uses the modified Hall Method in toroid coordinates
        z = x * (2 * RS / (G + 1))**0.5    # Eq. 12 in Hall paper
        RSP = RS + 1
        u[1] = y*y/2 - 0.25 + z
        v[1] = y*y*y/4 - y/4 + y*z
        u[2] = (2*G + 9)*y*y*y*y/24 - (4*G + 15)*y*y/24 + (10*G + 57)/288 + z*(y*y - 5/8) - (2*G - 3)*z*z/6
        v[2] = (G + 3)*y*y*y*y*y/9 - (20*G + 63)*y*y*y/96 + (28*G + 93)*y/288 + z*((2*G + 9)*y*y*y/6 - (4*G + 15)*y/12) + y*z*z
        u[3] = (556*G*G + 1737*G + 3069)*y*y*y*y*y*y/10368 - (388*G*G + 1161*G + 1881)*y*y*y*y/2304 + (304*G*G + 831*G + 1242)*y*y/1728 - (2708*G*G + 7839*G + 14211)/82944 + z*((52*G*G + 51*G + 327)*y*y*y*y/34 - (52*G*G + 75*G + 279)*y*y/192 + (92*G*G + 180*G + 639)/1152) + z*z*(-(7*G - 3)*y*y/8 + (13*G - 27)/48) + (4*G*G - 57*G + 27)*z*z*z/144
        v[3] = (6836*G*G + 23031*G + 30627)*y*y*y*y*y*y*y/82944 - (3380*G*G + 11391*G + 15291)*y*y*y*y*y/13824 + (3424*G*G + 11271*G + 15228)*y*y*y/13824 - (7100*G*G + 22311*G + 30249)*y/82944 + z*((556*G*G + 1737*G + 3069)*y*y*y*y*y/1728 * (388*G*G + 1161*G + 1181)*y*y/576 + (304*G*G + 831*G + 1242)*y/864) + z*z*((52*G*G + 51*G + 327)*y*y*y/192 - (52*G*G + 75*G + 279)*y/192) - z*z*z*(7*G - 3)*y/12
        
        U = 1 + u[1]/RSP + (u[1] + u[2])/(RSP*RSP) + (u[1] + 2*u[2] + u[3])/(RSP*RSP*RSP)
        V = ((G+1)/(2*RSP))*(v[1]/RSP + (1.5*v[1] + v[2])/(RSP*RSP) + (15./8.*v[1] + 2.5*v[2] + v[3])/(RSP*RSP*RSP))**0.5
    
    elif (GEOM == 1):
        #  Calculate the z to be used in the velocity equations Eq 12
        z = x * (RS / (G + 1))**0.5

        u[1] = 0.5*y*y - 1/6 + z
        v[1] = y*y*y/6 - y/6 + y*z
        u[2] = (y+6)*y*y*y*y/18 - (2*G+9)*y*y/18 + (G+30)/270 + z*(y*y-0.5) - (2*G-3)*z*z/6
        v[2] = (22*G+75)*y*y*y*y*y/360 - (5*G+21)*y*y*y/54 + (34*G+195)*y/1080 + z/9*((2*G+12)*y*y*y - (2*G+9)*y) + y*z*z
        u[3] = (362*G*G+1449*G+3177)*y*y*y*y*y*y/12960 - (194*G*G + 837*G + 1665)*y*y*y*y/2592 + (854*G*G + 3687*G + 6759)*y*y/12960 - (782*G*G + 5523 + 2*G*2887)/272160 + z*((26*G*G + 27*G + 237)*y*y*y*y/288 - (26*G*G + 51*G + 189)*y*y/144 + (134*G*G + 429*G + 1743)/4320) + z*z*(-5*G*y*y/4 + (7*G - 18)/36) + z*z*z*(2*G*G - 33*G + 9)/72
        v[3] = (6574*G*G + 26481*G + 40059)*y*y*y*y*y*y*y/181440 - (2254*G*G + 10113*G + 16479)*y*y*y*y*y/25920 + (5026*G*G + 25551*G + 46377)*y*y*y/77760 - (7570*G*G + 45927*G + 98757)*y/544320 +  z*((362*G*G + 1449*G + 3177)*y*y*y*y*y/2160 * (194*G*G + 837*G + 1665)*y*y*y/648 + (854*G*G + 3687*G + 6759)*y/6480) + z*z*((26*G*G + 27*G + 237)*y*y*y/144 - (26*G*G + 51*G + 189)/144) + z*z*z*(-5*G*y/6)
        
        U = 1 + u[1]/RS + u[2]/RS/RS + u[3]/RS/RS/RS
        V = ((G+1)/RS)*(v[1]/RS + v[2]/RS/RS + v[3]/RS/RS/RS)**0.5

    if (abs(V) < 1e-5): 
        V = 0.0

    _theta = math.atan2(V,U)

    if (abs(_theta) < 1e-5):
        _theta = 0.0

    _ma = (U * U + V * V)**0.5

    if _ma < 1.0:
        raise ValueError("Intial Data Line is subsonic, try increasing Initial Line Angle")

    return _theta, _ma


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

    n = 12

    ktta = 8.0
    upperwall = WallPoints()
    upperwall.add_section(np.sin(np.linspace(0., math.pi / 180. * ktta, 9)), lambda x: 8. - (6.**2 - x**2)**0.5)
    upperwall.add_section(np.linspace(0, 5, 12), lambda x: math.tan(math.pi / 180. * ktta) * x)
    # upperwall.plot()

    # x0 = np.zeros(n)
    # y0 = np.linspace(2., 0., n)
    # print(len(y0))
    # tta0 = np.zeros(n)
    # p0 = np.ones(n) * 101325
    # rho0 = np.ones(n) * 1.125
    # vel0 = np.ones(n) * (1.4 * p0 / rho0)**0.5 * 1.1
    # init_line = [Node(x0[i], y0[i], rho0[i], p0[i], vel0[i], tta0[i]) for i in range(n)]
    init_line = calc_initial_throat_line(n, 2.0, rUp=9., pTotal=2015., tTotal=2726.)
    # plt.plot([pt.x for pt in init_line], [pt.y for pt in init_line], '-o')
    # plt.plot([pt.x for pt in init_line], [pt.ma for pt in init_line], '-o')
    # plt.show()

    sym_node = copy.deepcopy(init_line[-1])

    grid_points = []
    grid_points.append(init_line)

    step = 0
    max_step = 3
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