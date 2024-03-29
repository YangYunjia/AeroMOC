from .utils import *
import numpy as np
import math
from typing import Tuple, List, Dict, Callable,TypeVar


class BasicNode():
    '''
    This base class is for geometry properties of a node. 

    It only has coordinate (x, y) [**TODO** extend to 3-D].

    ### initilal:

    >>> p1 = BasicNode()         # without assign coordinate
    >>> p2 = BasicNode(1.0, 2.0) # assign coordinate

    ### properties:
    `x`, `y`
    
    '''
    def __init__(self, *cor) -> None:
        if len(cor) == 2:
            self.cors = np.array(cor)
        else:
            self.cors   = np.ones(2) * (-BIG_NUMBER)   # x, y
        # the origin coordinates of lrc and rrc, for ploting
        # to avoid slow, only the array is saved, not the point object
        self.streamnode: np.array = None
        self.leftnode: np.array = None
        self.rightnode: np.array = None    

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

    def lastnode(self, n, dirc: int):
        '''
        record last nodes

        ### para:
        - `n`:  (BasicNode)     the last node
        - `dirc`:   dirction

        '''
        if dirc == LEFTRC:
            self.leftnode = n.cors
        elif dirc == RIGHTRC:
            self.rightnode = n.cors
        else:
            self.streamnode = n.cors

class Node(BasicNode):
    '''
    This base class is for a normal node. 
    
    ### initilal:

    >>> p1 = BasicNode()         # without assign coordinate
    >>> p2 = BasicNode(1.0, 2.0) # assign coordinate

    ### properties:

    #### Independent properties (can be set from outside)
    - coordinate (`x`, `y`)
    - density (`rho`)
    - pressure (`p`)
    - velocity magnitude (`vel`)
    - velocity direction from x-dirction (`tta`)
    
    #### Others
    - mach angle (`alp`)  = 1 / sin(Ma)
    - characteristic angle (`lam`)
        - use indicator `p1.lam(dirc=RIGHTRC)` 
            - dirc = LEFTRC (+1) for left running
            - dirc = RIGHTRC (-1) for right runing
        - use `p1.lam_plus` (left) and `p1.lam_minus` (right)
    - acoustic speed (`a`)
    - mach number (`ma`)
    - temperature (`t`)

    #### Set the properties
    - directly set the independent properties
    - `set_by_total`:   set by the total pressure, temperature, mach number and flow angle
    - `set_by_static`:  set by the static pressure, temperature, mach number and flow angle
    '''


    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.vals   = np.array([-1.0, -1.0, -1.0, BIG_NUMBER])  # rho, p, vel, tta
        self.g      = 1.4

    def __repr__(self) -> str:
        return '(%.2f, %.2f) (%.2f, %.2f %.2f, %.2f)' % tuple(list(self.cors) + list(self.vals))

    def set_by_total(self, tta: float, ma: float, pt: float, tt: float) -> None:
        '''
        set by the total pressure, temperature, mach number and flow angle
        '''
        p, t, rho = calc_isentropicPTRHO(g=self.g, ma=ma, pTotal=pt, tTotal=tt)
        self.rho = rho
        self.p   = p
        self.vel = ma * (self.g * GAS_GAMMA * t)**0.5
        self.tta = tta

    def set_by_static(self, tta: float, ma: float, p: float, t: float) -> None:
        '''
        set by the static pressure, temperature, mach number and flow angle
        '''
        self.rho = p / (GAS_GAMMA * t)
        self.p   = p
        self.vel = ma * (self.g * GAS_GAMMA * t)**0.5
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
        if self.vals[3] < PI and self.vals[3] > -PI: 
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

    @property
    def t(self) -> float:
        return self.a**2 / self.g / GAS_GAMMA

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
        _betas = np.linspace(0, PI / 2., 1000)
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
    if ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) < 1e-10:
        return True
    else:
        return False