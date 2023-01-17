'''
MOC

YangYunjia, 2023.1.3

'''
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Callable

GEOM = 0
GAS_R = 1.4
BIG_NUMBER = 100000.

WALLTYP = ['wall']
SYMTYP  = ['sym']
UPP = ['u', 'upper']
LOW = ['l', 'lower']
LEFTRC = +1
RIGHTRC = -1

class ExtrapolateError(Exception):
    pass

class EndofwallError(Exception):
    pass

class SubsonicError(Exception):
    
    def __init__(self, *cors, info) -> None:
        super().__init__('Subsonic flow occurs during "%s" at point (x,y) = %.4f, %.4f' % (info, cors[0], cors[1]))

class KeySelectError(Exception):
    
    def __init__(self, key, value) -> None:
        super().__init__('No value: "%s" for key "%s"' % (value, key))

class BasicNode():
    def __init__(self) -> None:
        self.cors   = np.ones(2) * (-BIG_NUMBER)   # x, y

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

class Node(BasicNode):

    def __init__(self, *value) -> None:
        super().__init__()
        self.vals   = np.array([-1.0, -1.0, -1.0, BIG_NUMBER])  # rho, p, vel, tta
        
        if len(value) in [2, 6]:
            self.cors = np.array(value[:2])
        if len(value) == 6:
            self.vals = np.array(value[2:6])

        self.g      = 1.4

    def __repr__(self) -> str:
        return '(%.2f, %.2f) (%.2f, %.2f %.2f, %.2f)' % tuple(list(self.cors) + list(self.vals))

    def set_by_total(self, tta: float, ma: float, pt: float, tt: float) -> None:
        p, t, rho = calc_isentropicPTRHO(g=self.g, ma=ma, pTotal=pt, tTotal=tt)
        self.rho = rho
        self.p   = p
        self.vel = ma * (self.g * GAS_R * t)**0.5
        self.tta = tta

    def set_by_static(self, tta: float, ma: float, p: float, t: float) -> None:
        self.rho = p / (GAS_R * t)
        self.p   = p
        self.vel = ma * (self.g * GAS_R * t)**0.5
        self.tta = tta

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
    
    # TODO: lam_plus and lam_minus should be closed to lam()
    def lam(self, dirc: int) -> float:
        '''
        calculate the characteristic angle for Characteristic from this node(positive)
        (dirc = positive 1 for left running; dirc = negative 1 for right runing)
        '''
        return math.tan(self.tta + dirc * self.alp)

    @property
    def lam_plus(self) -> float:
        '''
        calculate the characteristic angle for Left Runing Characteristic from this node (positive)
        '''
        return math.tan(self.tta + self.alp)

    @property
    def lam_minus(self) -> float:
        '''
        calculate the characteristic angle for Left Runing Characteristic from this node(negative)
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
        _ma = self.vel / self.a
        if _ma >= 1.0: 
            return _ma
        else:
            raise ValueError('Value for ma = %.3f is not correct' % _ma)

class ShockNode(BasicNode):
    
    def __init__(self, nodef: Node) -> None:
        super().__init__()
        self.cors = np.array([nodef.x, nodef.y])   # x, y
        
        self.nf   = nodef
        self.nb   = Node(nodef.x, nodef.y)
        self.ttas = BIG_NUMBER

    def set_by_ttab(self, ttab: float):
        self.nb.tta = ttab
        _delta     = self.nb.tta - self.nf.tta   # deflection angle
        _tan_delta = math.tan(_delta)
        _mach  = self.nf.ma
        _g     = self.nf.g
        # use a table to find beta
        _betas = np.linspace(0, math.pi / 2., 1000)
        _tan_deltas = 2. / np.tan(_betas) * (_mach**2 * np.sin(_betas)**2 - 1.) / (_mach**2 * (_g + np.cos(2 * _betas) + 2.))
        for _i in range(len(_tan_deltas)):
            if _tan_delta <= _tan_deltas[_i] and _tan_delta > _tan_deltas[_i-1]:
                _beta = _betas[_i-1] + (_tan_delta - _tan_deltas[_i-1]) / (_tan_deltas[_i] - _tan_deltas[_i-1]) * (_betas[_i] - _betas[_i-1])
                break
        else:
            raise SubsonicError(self.cors, info='Too high deflection angle to form a detached shock wave')
        
        self.ttas = self.nf.tta + _beta
        _ma1sb2 = _mach**2 * math.sin(_beta)**2
        self.nb.rho = self.nf.rho * ((_g + 1) * _ma1sb2 / (2 + (_g - 1) * _ma1sb2))
        self.nb.p   = self.nf.p   * (2. * _g / (_g + 1) * _ma1sb2 - (_g - 1) / (_g + 1))
        self.nb.vel = self.nf.vel * (math.cos(_beta) / math.cos(_beta - _delta))

        _mach2 = self.nb.ma
        if _mach2 <= 1.0:
            raise SubsonicError(self.cors, info='Subsonic (Ma2 = %.4f) after oblique shoch wave' % _mach2)

def same_node(p1: Node, p2: Node) -> bool:
    if ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) < 1e-6:
        return True
    else:
        return False


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



def calc_isentropicPTRHO(g: float, ma: float, pTotal: float, tTotal: float) -> Tuple[float, float, float]:
    ratio = 1 + (g - 1) / 2. * ma**2
    p = pTotal / ratio**(g / (g - 1))
    t = tTotal / ratio
    rho = p / (GAS_R * t)
    return p, t, rho

def _calc_p3_xy(p1x: float, p1y: float, p2x: float, p2y: float, p4x: float, p4y: float, p4tta: float) -> float:
    '''
    calculate the ratio (3 - 2) / (1 - 2) given the coordinate of p.1, p.2 and p.4 and the absolute angle 3-4

    >>>    1 -
    >>>    |      -  4
    >>>    3 --    -
    >>>    |   -
    >>>    2-

    '''
    if abs((p1y - p2y - p4tta * (p1x - p2x))) < 0.001:
        raise ExtrapolateError()
        # pass
        # print('!')
        # plt.show()
    return (p4y - p2y - p4tta * (p4x - p2x)) / (p1y - p2y - p4tta * (p1x - p2x))

def _calc_p4_xy(p1x: float, p1y: float, p1tta: float, p2x: float, p2y: float, p2tta: float) -> Tuple[float, float]:
    p4x = ((p1y - p2y) - (p1tta * p1x - p2tta * p2x)) / (p2tta - p1tta)
    p4y = p2y + p2tta * (p4x - p2x)
    return p4x, p4y

def _calc_p4_vals(_p1: Node, _p2: Node, _p4: Node) -> bool:
    _p3 = Node()

    T_plus  = _p2.Q * _p2.p + _p2.tta - _p2.S_plus  * (_p4.x - _p2.x)
    T_minus = _p1.Q * _p1.p - _p1.tta - _p1.S_minus * (_p4.x - _p1.x)
    _p4.p = (T_plus + T_minus) / (_p1.Q + _p2.Q)
    _p4.tta = T_plus - _p2.Q * _p4.p

    # the ratio of point 3 on the line p1-p2 from p2
    ratio = _calc_p3_xy(_p1.x, _p1.y, _p2.x, _p2.y, _p4.x, _p4.y, _p4.tta)

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
    _p1 = copy.deepcopy(p1)
    _p2 = copy.deepcopy(p2)

    p4x, p4y = _calc_p4_xy(_p1.x, _p1.y, _p1.lam_minus, _p2.x, _p2.y, _p2.lam_plus)
    p4  = Node(p4x, p4y)

    # if new node is too close to the p1 or p2, return old node
    if same_node(_p1, p4):
        return p1
    if same_node(_p2, p4):
        return p2

    # predict step
    _calc_p4_vals(_p1, _p2, p4)
    # correction step
    _p1.vals = 0.5 * (_p1.vals + p4.vals)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _calc_p4_vals(_p1, _p2, p4)

    plt.plot([p1.x, p4.x], [p1.y, p4.y], '-', c='b')
    plt.plot([p2.x, p4.x], [p2.y, p4.y], '-', c='r')

    return p4

def calc_wall_point(xx: float, yy: float, dydx: float, last_line: List[Node or ShockNode], dirc: int) -> Tuple[Node, int]:
    
    p4  =  Node(xx, yy)
    p4.tta = math.atan(dydx)
    p2 = Node()
    llidx = 0   # node index on the last rrc line

    while llidx < len(last_line) - 1:
    
        _p5 = copy.deepcopy(last_line[llidx])
        # remind that only can the last point of the last rrc line be a ShockNode
        if isinstance(last_line[llidx + 1], ShockNode):
            _p3 = copy.deepcopy(last_line[llidx + 1].nb)
        else:
            _p3 = copy.deepcopy(last_line[llidx + 1])

        ratio = 0.5
        ratio_old = 0.0
        # decide the point2 (origin of the LRC that ends at point4)
        try:
            while abs(ratio - ratio_old) > 1e-3:
                ratio_old = ratio
                p2.cors = _p5.cors + ratio * (_p3.cors - _p5.cors)
                p2.vals = _p5.vals + ratio * (_p3.vals - _p5.vals)
                ratio = _calc_p3_xy(_p3.x, _p3.y, _p5.x, _p5.y, p4.x, p4.y, p2.lam(-dirc))
                if ratio > 1.0 or ratio < 0.0: raise ExtrapolateError()
            break
        except ExtrapolateError:
            llidx += 1
    
    else:
        # when can find a interaction point between the lrc to the wall point, and the last rrc
        # two possible condition:
        #  - first: the last rrc is intercepted by an shock wave 
        #  - second: the solution is fail
        if isinstance(last_line[llidx + 1], ShockNode):
            # p.2 is obtained by find the intersection betw. s.w and lrc
            raise NotImplementedError()

    _p2 = copy.deepcopy(p2)
    if dirc == RIGHTRC:
        _calc_boundary_p4_vals(_p3, p4, _p2=_p2)
    else:
        _calc_boundary_p4_vals(_p3, p4, _p1=_p2)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _p3.vals = 0.5 * (_p3.vals + p4.vals)
    if dirc == RIGHTRC:
        _calc_boundary_p4_vals(_p3, p4, _p2=_p2)
    else:
        _calc_boundary_p4_vals(_p3, p4, _p1=_p2)

    plt.plot([last_line[0].x, p4.x], [last_line[0].y, p4.y], '-', c='k')
    plt.plot([p2.x,           p4.x], [p2.y,           p4.y], '-', c='r')

    return p4, llidx

def calc_sym_point(p1: Node, p3: Node, dirc: int) -> Node:

    p4 = Node()
    _p1 = copy.deepcopy(p1)
    _p3 = copy.deepcopy(p3)

    p4.y = 0.0  # TODO
    p4.x = p1.x + (p4.y - p1.y) / p1.lam(dirc)

    p4.tta = 0.0
    
    if dirc == RIGHTRC:
        _calc_boundary_p4_vals(_p3, p4, _p1=_p1)
    else:
        _calc_boundary_p4_vals(_p3, p4, _p2=_p1)
    _p1.vals = 0.5 * (_p1.vals + p4.vals)
    if dirc == RIGHTRC:
        _calc_boundary_p4_vals(_p3, p4, _p1=_p1)
    else:
        _calc_boundary_p4_vals(_p3, p4, _p2=_p1)

    plt.plot([p3.x, p4.x], [p3.y, p4.y], '--', c='k')
    plt.plot([p1.x, p4.x], [p1.y, p4.y], '-', c='b')

    return p4

def calc_shock_wall_point(xx: float, yy: float, dydx1: float, dydx2: float, last_line: List[Node]) -> Tuple[ShockNode, int]:
    wall_point, _i = calc_wall_point(xx, yy, dydx1, last_line)
    wall_node = ShockNode(nodef=wall_point)
    wall_node.set_by_ttab(ttab=dydx2)
    return wall_node, _i

def calc_shock_interior_point(p0: ShockNode, p2: Node) -> ShockNode:
    
    _p0 = copy.deepcopy(p0)
    _p2 = copy.deepcopy(p2)

    # (1) determin the x,y of the p.4
    p4x, p4y = _calc_p4_xy(_p0.x, _p0.y, _p0.ttas, _p2.x, _p2.y, _p2.lam_plus)
    if p4x <= p2.x:     # shock wave interact with the current rrc
        raise ExtrapolateError()
    p4  = Node(p4x, p4y)
    
    # (2) find the p.1 (inital point of the lrc to p.4)
    p1 = Node()
    p1ratio = 0.5
    p1ratio_old = 0.0
    while abs(p1ratio - p1ratio_old) > 1e-3:
        p1ratio_old = p1ratio
        p1.cors = _p2.cors + p1ratio * (_p0.cors - _p2.cors)
        p1.vals = _p2.vals + p1ratio * (_p0.nf.vals - _p2.vals)
        p1ratio = _calc_p3_xy(_p0.x, _p0.y, _p2.x, _p2.y, p4x, p4y, p1.lam_minus)
        if p1ratio > 1.0 or p1ratio < 0.0: raise ExtrapolateError()

    # (3) calculate the p.4 with interior point formula
    # predict step
    _calc_p4_vals(p1, _p2, p4)
    # correction step
    p1.vals  = 0.5 * (p1.vals  + p4.vals)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _calc_p4_vals(p1, _p2, p4)
    


    pass


def calc_throat_point(_yy: float, last_line: List[Node],
                        mode: str, rUp: float, p: float, t: float, mT: float, 
                        lmmax: float, lmmin: float) -> Node:

    if len(last_line) > 0:
        _xx_old = last_line[-1].x
        _yy_old = last_line[-1].y
        dydx = last_line[-1].lam_minus   # this calculates a new X first assumed it falls on the RRC

    
    _mach = 0.0
    point = Node()
    
    if mode in ['total']:
        ratio = 1.0
        if len(last_line) > 0:
            while ratio >= 1.0:
                _xx = _xx_old + (_yy - _yy_old) / (dydx * ratio)
                _theta, _mach = KLThroat(_xx, _yy, point.g, rUp)
                # increase dydx so that slope is 1.1 of what is was, only when mach number exceed 1.5
                # this is to prevent the initial line is too steep and lead to solution failure
                if _mach > lmmax:
                    ratio *= 1.1
                elif _mach < lmmin:
                    ratio /= 1.1
                else:
                    break
                # print(_mach, dydx)
        else:
            _xx = 0.0
            _theta, _mach = KLThroat(_xx, _yy, point.g, rUp)

        point.set_by_total(tta=_theta, ma=_mach, pt=p, tt=t)
        
    elif mode in ['static']:
        _theta = 0.0
        _mach  = mT
        if len(last_line) > 0:
            _xx = _xx_old + (_yy - _yy_old) / dydx
        else:
            _xx = 0.0

        point.set_by_static(tta=0.0, ma=mT, p=p, t=t)

    else:
        raise KeySelectError('initial line mode', mode)

    if point.ma < 1.0:
        raise SubsonicError(point.x, point.y, info="Calculated Throat Mach number < 1.0 at CalcInitialThroatLine")

    point.x = _xx
    point.y = _yy

    # print(i, _theta, _mach)
    return point

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
        V = ((G+1)/(2*RSP))**0.5 * (v[1]/RSP + (1.5*v[1] + v[2])/(RSP*RSP) + (15./8.*v[1] + 2.5*v[2] + v[3])/(RSP*RSP*RSP))
    
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
        V = ((G+1)/RS)**0.5 * (v[1]/RS + v[2]/RS/RS + v[3]/RS/RS/RS)

    if (abs(V) < 1e-5): 
        V = 0.0

    _theta = math.atan2(V,U)
    # print(U, V, _theta)

    if (abs(_theta) < 1e-5):
        _theta = 0.0

    _ma = (U * U + V * V)**0.5

    # if _ma < 1.0:
    #     raise ValueError("Intial Data Line is subsonic, try increasing Initial Line Angle")

    return _theta, _ma


class MOC2D():
    '''
    main class
    '''

    def __init__(self) -> None:
        self.utyp: str = None
        self.ltyp: str = None
        self.uy0 = 1.0
        self.ly0 = 0.0
        self.upoints: WallPoints = None
        self.lpoints: WallPoints = None
        self.urUp = 0.0
        self.lrUp = 0.0

        self.rrcs: List[List[Node]] = []
        self.lrcs: List[List[Node]] = []

    def set_boundary(self, side: str, typ: str, y0: float, points: WallPoints = None, rUp: float = 0.0) -> None:

        if side in UPP:
            self.utyp = typ
            self.uy0  = y0
            self.upoints = points
            self.urUp = rUp
        elif side in LOW:
            self.ltyp = typ
            self.ly0  = y0
            self.lpoints = points 
            self.lrUp = rUp           

    def _dimen(self, ip: Node, dirc: int) -> Node:
        
        newp = copy.deepcopy(ip)
        newp.x = (self.uy0 - self.ly0) * ip.x
        if dirc == RIGHTRC:
            newp.y = self.ly0 + (self.uy0 - self.ly0) * ip.y
        else:
            newp.y = self.uy0 - (self.uy0 - self.ly0) * ip.y
            newp.tta = -ip.tta

        return newp


    def calc_initial_throat_line(self, n: int, mode: str = 'total', p: float = 0.0, t: float = 0.0, mT: float = 0.0,
                **para: Dict):
        '''
        This function is interpreted from code of CalcInitialThroatLine <- MOC_GidCalc_BDE <- the MOC programma of NASA
        '''
        
        if 'LineMaMax' in para.keys():
            lmmax = para['lineMaMax']
        else:
            lmmax = 1.5
        if 'LineMaMin' in para.keys():
            lmmin = para['lineMaMin']
        else:
            lmmin = 1.01
        
        rline: List[Node] = []
        lline: List[Node] = []
        nrline: List[Node] = []
        nlline: List[Node] = []

        # determine the y-dirction points, input yy should be non-dimensional
        if 'InitPointsDistri' in para.keys():
            if para['InitPointsDistri'] in ['equal']:
                _yy = (n - np.arange(n + 1)) / n
            elif para['InitPointsDistri'] in ['sin']:
                pows =  1.5
                _yy = np.sin(math.pi * (n - np.arange(n + 1)) / n / 2.)**pows
                # Based on the way it is done in RAO, the y[i] is based on a sinusoidal
                # and x[i] is assumed to be on the RRC from the throat wall
            else:
                raise KeySelectError('InitPointsDistri', para['InitPointsDistri'])
        else:
            _yy = (n - np.arange(n + 1)) / n

        ir = 0
        il = n
        while ir <= n and il >= 0:

            # two Characteristic line origins from both side, and find the intersection point
            if self.utyp in WALLTYP:
                npoint = calc_throat_point(_yy[ir], nrline, mode, self.urUp, p, t, mT, lmmax, lmmin)
                nrline.append(npoint)
                rline.append(self._dimen(npoint, RIGHTRC))

                if len(lline) >= (n-(ir-1)+1) and (rline[ir-1].x - lline[n-(ir-1)].x) * (rline[ir].x - lline[n-ir].x) <= 0.0:
                    cpoint = calc_interior_point(rline[ir-1], lline[n-ir])
                    lline = lline[:(n-ir)+1] + [cpoint]
                    rline = rline[:(ir-1)+1] + [cpoint]
                    break
                
                ir += 1

            if self.ltyp in WALLTYP:
                npoint = calc_throat_point(1.-_yy[il], nlline, mode, self.lrUp, p, t, mT, lmmax, lmmin)
                nlline.append(npoint)
                lline.append(self._dimen(npoint, LEFTRC))

                if len(rline) >= ((il+1)+1) and (rline[il].x - lline[n-il].x) * (rline[il+1].x - lline[n-(il+1)].x) <= 0.0:
                    # left line intersect into existing left line
                    cpoint = calc_interior_point(rline[il], lline[n-(il+1)])
                    lline = lline[:(n-(il+1))+1] + [cpoint]
                    rline = rline[:il        +1] + [cpoint]
                    break

                il -= 1

        plt.plot([ip.x for ip in rline], [ip.y for ip in rline], '-x', c='b')
        plt.plot([ip.x for ip in lline], [ip.y for ip in lline], '-x', c='r')

        self.lrcs.append(lline)
        self.rrcs.append(rline)

    def calc_chara_line(self):

        newrrc: List[Node] = []
        newlrc: List[Node] = []

        # upper wall
        if self.utyp in WALLTYP:
            try:
                lastrrc = self.rrcs[-1]
                _xw, _yw, _dydxw = next(self.upoints)
                wall_point, _i = calc_wall_point(_xw, _yw, _dydxw, lastrrc, dirc=RIGHTRC)
                newrrc.append(wall_point)
                
                # calculate interior points amount = (N_lastline - 1)
                for _ii in range(_i + 1, len(lastrrc)):
                    newrrc.append(calc_interior_point(newrrc[-1], lastrrc[_ii]))

                if self.ltyp in SYMTYP:
                    # print(len(newrrc), len(lastrrc))
                    newrrc.append(calc_sym_point(newrrc[-1], lastrrc[-1], dirc=RIGHTRC))

            except EndofwallError:
                pass
        
        if self.ltyp in WALLTYP:
            try:
                lastlrc = self.lrcs[-1]
                _xw, _yw, _dydxw = next(self.lpoints)
                wall_point, _i = calc_wall_point(_xw, _yw, _dydxw, lastlrc, dirc=LEFTRC)
                newlrc.append(wall_point)
                
                # calculate interior points amount = (N_lastline - 1)
                for _ii in range(_i + 1, len(lastlrc)):
                    newlrc.append(calc_interior_point(lastlrc[_ii], newlrc[-1]))

                if self.utyp in SYMTYP:
                    # print(len(newrrc), len(lastrrc))
                    newlrc.append(calc_sym_point(newlrc[-1], lastlrc[-1], dirc=LEFTRC))

            except EndofwallError:
                pass

        if len(newlrc) > 0 and len(newrrc) > 0:
            cpoint = calc_interior_point(newrrc[-1], newlrc[-1])
            newlrc.append(cpoint)
            newrrc.append(cpoint)

        self.lrcs.append(newlrc)
        self.rrcs.append(newrrc)

    def solve(self, max_step: int = 100000):
        
        step = 0

        while step < max_step and (len(self.lrcs[-1]) > 0 or len(self.rrcs[-1]) > 0):

            self.calc_chara_line()
            step += 1

    def plot_wall(self, side, var='p'):
        if side in UPP: 
            plt.plot(range(len(self.rrcs[:-1])), [l[0].p for l in self.rrcs[:-1]])
        if side in LOW: 
            plt.plot(range(len(self.lrcs[:-1])), [l[0].p for l in self.lrcs[:-1]])


    def calc_shock_line(self, _lines: List[List[Node]], _xw: float, _yw: float, _dydxw1: float, _dydxw2: float) -> List[ShockNode]:

        # calculate shock node on the initial point (A.)
        wall_point, _i = calc_shock_wall_point(_xw, _yw, _dydxw1, _dydxw2, _lines[-1])
        new_line = [wall_point]

        raise NotImplementedError()


if __name__ == '__main__':

    eps = 1e-2

    n = 9

    ktta = 8.0
    upperwall = WallPoints()
    # upperwall.add_section(6 * np.sin(np.linspace(0., math.pi / 180. * ktta, 15)), lambda x: -4. + (6.**2 - x**2)**0.5)
    # upperwall.add_section(np.linspace(0, 5, 12), lambda x: -math.tan(math.pi / 180. * ktta) * x)
    upperwall.add_section(6 * np.sin(np.linspace(0., math.pi / 180. * ktta, 15)), lambda x: 8. - (6.**2 - x**2)**0.5)
    upperwall.add_section(np.linspace(0, 5, 12), lambda x: math.tan(math.pi / 180. * ktta) * x)
    # upperwall.plot()

    ktta = 12.0

    lowerwall = WallPoints()
    lowerwall.add_section(2 * np.sin(np.linspace(0., math.pi / 180. * ktta, 15)), lambda x: -4. + (2.**2 - x**2)**0.5)
    lowerwall.add_section(np.linspace(0, 5, 12), lambda x: -math.tan(math.pi / 180. * ktta) * x)
    
    moc = MOC2D()
    moc.set_boundary('u', typ='wall', y0=2.0, points=upperwall, rUp=9.)
    # moc.set_boundary('l', typ='sym',  y0=0.0)
    moc.set_boundary('l', typ='wall', y0=-2.0, points=lowerwall, rUp=3.)
    # moc.set_boundary('u', typ='sym',  y0=0.0)

    # init_line = calc_initial_throat_line(n, 2.0, mode='static', p=101325, t=283, mT=2.2)
    moc.calc_initial_throat_line(n, mode='total', p=2015., t=2726.)
    # plt.plot([pt.x for pt in init_line], [pt.y for pt in init_line], '-o', c='k')
    # plt.plot([pt.x for pt in init_line], [pt.ma for pt in init_line], '-o')
    # plt.show()

    moc.solve(max_step=30)

    # plt.xlim(0,1)
    plt.show()

    moc.plot_wall(side='l')

    # plt.plot(range(10), [grid_points[i][0].vel for i in range(10)])
    # plt.plot(range(10), [grid_points[i][0].a   for i in range(10)])
    # plt.plot(range(10), [grid_points[i][0].alp / 3.14 * 180 for i in range(10)])
    # plt.plot(range(10), [grid_points[i][0].tta / 3.14 * 180 for i in range(10)])
    plt.show()