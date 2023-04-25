import math
import copy
import numpy as np
from typing import Tuple, List

from .utils import *
from .node import Node, ShockNode, same_node

######################################
# functions for initial line
######################################

def calc_throat_point(_yy: float, last_line: List[Node],
                        mode: str, rUp: float, p: float, t: float, mT: float, 
                        lmmax: float, lmmin: float) -> Node:
    '''
    This function is used to generate new points on initial line

    ### para:
    - `_yy`:  y coordinate for the new point (NON-DIMENSIONAL)
    - `last_line`:    existing initial line
    - `mode` (= `total` or `static`):   whether the inlet `p` and `t` is total or static value
    - `rUp`:    upstream radius
    - `p`:      inlet pressure (total or static)
    - `t`:      inlet temperature (total or static)
    - `mT` (only `mode` == `static`):   inlet mach number
    - `lmmax` & `lmmin` (only `mode` == `total`):
        - upper and lower bound for the Mach number on the initial line
        - according to reference, the Ma on init_line should not exceed 1.5
        - ref. Rice, Tharen. 2003. “2D and 3D Method of Characteristic Tools for Complex Nozzle Development Final Report.” RTDC-TPS-48I. THE JOHNS HOPKINS UNIVERSITY.
    
    '''
    if len(last_line) > 0:
        _xx_old = last_line[-1].x
        _yy_old = last_line[-1].y
        dydx = last_line[-1].lam_minus   # this calculates a new X first assumed it falls on the RRC

    point = Node()
    
    if mode in ['total']:
        _mach = 0.0
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

    ### para:
    - `x` is the streamwise distance of the point (NON-DIMENSIONAL)
    - `y` is the radial distance of the point (NON-DIMENSIONAL)
    - `G` is gamma
    - `RS` is the upstream radius

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

######################################
# functions for decide value of points
######################################

def _calc_p3_xy(p1x: float, p1y: float, p2x: float, p2y: float, p4x: float, p4y: float, p4tta: float) -> float:
    '''
    calculate the ratio (3 - 2) / (1 - 2) given the coordinate of p.1, p.2 and p.4 and the absolute angle 3-4

    >>>    1 -
    >>>    |      -  4
    >>>    3 --    -
    >>>    |   -
    >>>    2-

    '''
    if abs((p1y - p2y - p4tta * (p1x - p2x))) < 1e-5:
        return 1.0

    return (p4y - p2y - p4tta * (p4x - p2x)) / (p1y - p2y - p4tta * (p1x - p2x))

def _calc_p3_from_last_line(last_line: List[Node], llidx0: int, didx: int, p4x: float, p4y: float, p4tta: float) -> Tuple[Node, int]:
    '''
    find p.3 from last c.l, where the s.l originated from ends at p.4.

    the gradient of s.l. 3-4 is decided by p4tta and not iterate
    '''

    p3 = Node()

    llidx = llidx0   # node index on the last rrc line

    while llidx >= 0 and llidx < len(last_line) - 1:
    
        _p50 = last_line[llidx]
        _p51 = last_line[llidx + 1]
        ratio = _calc_p3_xy(_p50.x, _p50.y, _p51.x, _p51.y, p4x, p4y, p4tta)
        if ratio < 1.0 and ratio >= 0.0:
            break

        llidx += didx
        # print(llidx)

    else:
        raise ExtrapolateError()

    p3.cors = _p51.cors + ratio * (_p50.cors - _p51.cors)
    p3.vals = _p51.vals + ratio * (_p50.vals - _p51.vals)

    return p3, llidx

def _calc_p2_from_last_line(last_line: List[Node], p4x: float, p4y: float, dirc: int,
                             llidx0: int = 0, didx: int = 1)  -> Tuple[Node, int]:
    '''
    find p.2 from last r.c.l, where the l.c.l originated from ends at p.4.
    - gauss a point between two neighbour points on last c.l: p.50 and p.51
    - interpolate field value at this point
    - get angle of c.l. of this point
    - use this angle to refresh p.2.'s position
    - if can't find, step to next section (p.51 and p.52)

    the gradient of l.s.l. 2-4 (`p2.lam(-dirc)`) is decided by p4tta and iterate until covergence
    '''

    p2 = Node()
    llidx = llidx0   # node index on the last rrc line
    while llidx >= 0 and llidx < len(last_line) - 1:

        _p50 = copy.deepcopy(last_line[llidx])
        # remind that only can the last point of the last rrc line be a ShockNode
        # if isinstance(last_line[llidx + 1], ShockNode):
        #     _p51 = copy.deepcopy(last_line[llidx + 1].nb)
        # else:
        _p51 = copy.deepcopy(last_line[llidx + 1])

        ratio = 0.5
        ratio_old = 0.0
        # decide the point2 (origin of the LRC that ends at point4)
        try:
            while abs(ratio - ratio_old) > 1e-3:
                ratio_old = ratio
                p2.cors = _p51.cors + ratio * (_p50.cors - _p51.cors)
                p2.vals = _p51.vals + ratio * (_p50.vals - _p51.vals)
                ratio = _calc_p3_xy(_p50.x, _p50.y, _p51.x, _p51.y, p4x, p4y, p2.lam(-dirc))
                if ratio > 1.01 or ratio < -0.01: raise ExtrapolateError()
            break
        except ExtrapolateError:
            llidx += didx
    
    else:
        # when can't find a interaction point between the lrc to the wall point, and the last rrc
        # two possible condition:
        #  - first: the last rrc is intercepted by an shock wave 
        #  - second: the solution is fail
        if isinstance(last_line[llidx], ShockNode):
            # p.2 is obtained by find the intersection betw. s.w and lrc
            raise NotImplementedError()
        else:
            pass
    
    return p2, llidx

def _calc_p4_xy(p1x: float, p1y: float, p1tta: float, p2x: float, p2y: float, p2tta: float) -> Tuple[float, float]:
    '''
    calculate the position of p.4. from the [position and angle] of p.1 and p.2

    >>>    1 -
    >>>    |       -  4
    >>>    |     -
    >>>    |   -
    >>>    2-

    '''
    p4x = ((p1y - p2y) - (p1tta * p1x - p2tta * p2x)) / (p2tta - p1tta)
    p4y = p2y + p2tta * (p4x - p2x)
    return p4x, p4y

def _calc_p4_vals(_p1: Node, _p2: Node, _p4: Node, _p3: Node = None, last_line: List[Node] = None, llidx: int = 0) -> int:
    '''
    Calculate values of p.4. (p, theta, vel, rho). (p, theta) can be obtained by p.1. and p.2 from two c.l, 
    while (vel, rho) should be obtained by streamline (p.3. to p.4.) conservation.

    There are three ways to obtain p.3. :
    - directly give p.3.

    >>>     _ = _calc_p4_vals(p1, p2, p4, _p3=p3)

    - when calculating normal MOC or kernal region of nozzle, p.3. must located between p.1. and p.2., the p.3. 
    can be obtained by p.1. and p.2.

    >>>     _ = _calc_p4_vals(p1, p2, p4)

    - when calculating expansion region of nozzle, p.3., one of p.1. or p.2. is obtained backward. Thus where p.3.
    is located on the last c.l. is not obvious. It should be searched on the last c.l.
        - the start point for searching is input by `llidx`
        - `llidx` is decreased by 1 during searching
        - the place p.3. searched on last c.l. is returned
    
    >>>     new_llindx = _calc_p4_vals(p1, p2, p4, last_line=ll, llidx=idx)

    >>>     
    
    '''
    T_plus  = _p2.Q * _p2.p + _p2.tta - _p2.S_plus  * (_p4.x - _p2.x)
    T_minus = _p1.Q * _p1.p - _p1.tta - _p1.S_minus * (_p4.x - _p1.x)
    _p4.p = (T_plus + T_minus) / (_p1.Q + _p2.Q)
    _p4.tta = T_plus - _p2.Q * _p4.p

    if _p3 is not None:
        pass
    elif last_line is not None:
        _p3, llidx = _calc_p3_from_last_line(last_line, llidx, -1, _p4.x, _p4.y, _p4.tta)
    else:
        # the point 3 is obtained from p.1 and p.2
        ratio = _calc_p3_xy(_p1.x, _p1.y, _p2.x, _p2.y, _p4.x, _p4.y, _p4.tta)        # the ratio of point 3 on the line p1-p2 from p2
        _p3 = Node()
        _p3.vals = _p2.vals + ratio * (_p1.vals - _p2.vals)

    _p4.vel = _p3.vel - (_p4.p - _p3.p) / (_p3.rho * _p3.vel)
    _p4.rho = _p3.rho + (_p4.p - _p3.p) / (_p3.a**2)

    return llidx

def _calc_boundary_p4_vals(_p3: Node, _p4: Node, _p1: Node = None, _p2: Node = None) -> bool:
    '''
    Calculate the value for p4 from a characteristic line and a stream line

        p3: the origin of the s.l, `vel` , `rho` should be known at p3\n
        p4: the next interaction of c.l and s.l, the `x`, `y`, `tta` should be known at p4\n
        p1(rrc) or p2(lrc): the origin of the c.l, all value should be known at p1 or p2
    '''
    
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

######################################
# functions for generate new points
######################################

def calc_interior_point(p1: Node, p2: Node, last_line: List[Node] = None, llidx: int = 0) -> Node or Tuple[Node, int]:
    '''
    generate new interior point location and values. It can be 
    - forward (calculating normal MOC or kernal region of nozzle): p.1. and p.2. are
    on the last c.l., p.3. is located between p.1. and p.2.
    - backward (calculating expansion region of nozzle): one of p.1. or p.2. is on the 
    current c.l.. Thus p.3.'s location is not obvious.

    ### paras:
    - `p1`, `p2`:    the origins of the c.l, all values should be prescribed
        it doesn't matter p.4 get from p1, p2 by forward or backward
    
    #### for calculating backward:
    - `last_line`:     when p4 is get from backward, the location of p4
    is get from p1 and p2 by c.l, the rho, vel is get from find a p3 on the last
    c.l. When `last_line` is prescribed, the code automatically see p4 as backward
    and to find the p3 on the prescribed last c.l.
    - `llidx`:    (default is 0) the idx on the last line to start search with

    ### returns:
    - `p4`
    #### for calculating backward:
    - (`p4`, `llidx`)     (else)
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
    _ = _calc_p4_vals(_p1, _p2, p4, last_line=last_line, llidx=llidx)
    
    # correction step
    _p1.vals = 0.5 * (_p1.vals + p4.vals)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _llidx = _calc_p4_vals(_p1, _p2, p4, last_line=last_line, llidx=llidx)

    # print(p1)
    p4.lastnode(p1, dirc=RIGHTRC)
    p4.lastnode(p2, dirc=LEFTRC)
    # print(llidx, _llidx, last_line)
    if last_line is None:
        return p4
    else:
        return p4, _llidx

def calc_backwall_point(p3: Node, p7: Node, last_line: List[Node or ShockNode], dirc: int) -> Tuple[Node, int]:
    '''
    This method calculate nozzle contour wall points. There are several ways to 
    decide wall points
    - wave attenuation:     the dirction of s.l. 3-4 is decided by velocity 
    direction at p.3.

        new wall point's position is the intersection between s.l. 3-4 and backward
        c.l. 7-4. Then p.4.'s value can be decided by p.2., p.7., and p.3.

    - mass conservation: (TODO)


    >>>   (3) ==wall== (4)
    >>>    |        -    |
    >>>  (50)     -       |
    >>>    |    -        - (7) 
    >>>   (2) -      --      |
    >>>    |     --           |
    >>>  (51) -

    '''

    _p7 = copy.deepcopy(p7)
    xx, yy = _calc_p4_xy(p3.x, p3.y, p3.tta, _p7.x, _p7.y, _p7.lam(dirc=dirc))
    p2, _llidx = _calc_p2_from_last_line(last_line, xx, yy, dirc=dirc)
    p4  =  Node(xx, yy)
    _p2 = copy.deepcopy(p2)

    # 未知壁面点，从p.7.反推回来
    _ = _calc_p4_vals(_p7, _p2, p4, _p3=p3)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _p7.vals = 0.5 * (_p7.vals + p4.vals)       # new added
    _ = _calc_p4_vals(_p7, _p2, p4, _p3=p3)
    
    p4.lastnode(p7, dirc=dirc)
    p4.lastnode(p2, dirc=-dirc)
    p4.lastnode(p3, dirc=0)

    return p4, _llidx
    
def calc_boundary_point(mode: str, dirc: int, last_line: List[Node or ShockNode] = None,
                         xx: float = None, yy: float = None, dydx: float = 0.0,
                           p2: Node = None, p3: Node = None) -> Tuple[Node, int]:
    '''
    generate new boundary point. This means that only one origin of characteristic line
    is known. So more information (like the position of p.4.) should be prescribed.

    - `dirc`:    the direction of c.l. 3-2

    ### mode = wall
    new wall point (p.4.)'s xx, yy, and dydx are prescribed. Since the wall 
    is a streamline, p.3. is last wall point. p.2 is searched on the last c.l with
    `_calc_p2_from_last_line`. after position of p.2. is decided, variables on p.4. 
    can be decided by p.2. and p.3. (note that theta for p.4. is prescribed)
    
    prescribed variables:
    - p3, xx, yy, dydx, last_line

    >>>   (3) ==wall== (4)
    >>>    |          - 
    >>>  (50)       -
    >>>    |     - 
    >>>   (2) - 
    >>>    | 
    >>>  (51)

    ### mode = free
    It is a bit like mode:wall, the new point's flow angle theta `tta` is known. 
    The difference is thatthe position of p.4. is unknown, while the position of 
    p.2. is known. Thus p.4.'s position is first dicided by p.2. and p.3, then its
    another s.l. value are obtained by p.2.

    prescribed variables:
    - p2, p3

    >>>   (3) ==free== (4)
    >>>               | 
    >>>             | 
    >>>           |
    >>>       (2)

    ### mode = exit
    new wall points's xx, yy, dydx, and p.2. are prescribed. p.3. is searched on 
    the last_line

    >>>  (50) 
    >>>    | 
    >>>     (3) ------ (4)
    >>>      |        || 
    >>>      (51)    exit
    >>>        |   ||
    >>>         (2)

    '''
    llidx = 0

    if mode == 'wall':
        p2, llidx = _calc_p2_from_last_line(last_line, xx, yy, dirc=dirc)

    elif mode == 'free':
        xx, yy = _calc_p4_xy(p2.x, p2.y, p2.lam(dirc=-dirc), p3.x, p3.y, p3.tta)

    elif mode == 'exit':
        p3, llidx = _calc_p3_from_last_line(last_line, len(last_line) - 2, -1, xx, yy, dydx)
    else:
        raise KeySelectError('mode', mode)
    
    p4  =  Node(xx, yy)
    _p2 = copy.deepcopy(p2)

    # 已知壁面点
    p4.tta = dydx
    if dirc == RIGHTRC: _calc_boundary_p4_vals(p3, p4, _p2=_p2)
    else:               _calc_boundary_p4_vals(p3, p4, _p1=_p2)
    # 只有特征线相容方程采用Euler预估-校正法推进，流线等熵关系不采用预估-校正
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    if dirc == RIGHTRC: _calc_boundary_p4_vals(p3, p4, _p2=_p2)
    else:               _calc_boundary_p4_vals(p3, p4, _p1=_p2)

    p4.lastnode(p2, dirc=-dirc)
    p4.lastnode(p3, dirc=0)

    return p4, llidx

######################################
# functions for generate new lines
######################################

def calcback_charac_line(_xe: float, _ye: float, _extta: float, lastcl: List[Node], dirc:int):

    newcl: List[Node] = []
    try:
        exit_point, llidx = calc_boundary_point('exit', dirc, xx=_xe, yy=_ye, dydx=_extta, last_line=lastcl, p2=lastcl[-1])
        newcl.insert(0, exit_point)
    except ExtrapolateError:
        exit_point, _ = calc_boundary_point('free', dirc, p2=lastcl[-1], p3=lastcl[0], dydx=_extta)
        newcl.insert(0, exit_point)
        return newcl

    try:
        for _ii in range(len(lastcl) - 1, 0, -1):
            if dirc == RIGHTRC:
                # print('!', lastcl[0])
                newp, llidx = calc_interior_point(newcl[0], lastcl[_ii-1], last_line=lastcl, llidx=llidx)
                if newp.x > newcl[-1].x:
                    # 由于数值误差，上一条特征线倒数第二个点的左行特征线越过了exitplane，那么这个点不算了
                    continue
                newcl.insert(0, newp)
            else:
                raise NotImplementedError()
    except ExtrapolateError:
        # point 3 is out of last c.l, which means point 4 beyond wall
        pass

    p4, _ = calc_backwall_point(p3=lastcl[0], p7=newcl[0], last_line=lastcl, dirc=dirc)
    newcl.insert(0, p4)

    return newcl

def calc_charac_line(_xw: float, _yw: float, _dydxw: float, lastcl: List[Node], dirc: int) -> List[Node]:

    newcl: List[Node] = []
    wall_point, _i = calc_boundary_point('wall', dirc, p3=lastcl[0], last_line=lastcl, xx=_xw, yy=_yw, dydx=_dydxw)
    newcl.append(wall_point)
    
    # calculate interior points amount = (N_lastline - 1)
    for _ii in range(_i + 1, len(lastcl)):
        if dirc == RIGHTRC:
            newcl.append(calc_interior_point(newcl[-1], lastcl[_ii]))
        else:
            newcl.append(calc_interior_point(lastcl[_ii], newcl[-1]))

    return newcl

######################################
# functions for shock waves
######################################

def calc_shock_wall_point(xx: float, yy: float, dydx1: float, dydx2: float, last_line: List[Node]) -> Tuple[ShockNode, int]:
    raise NotImplementedError()
    # wall_point, _i = calc_wall_point(xx, yy, dydx1, last_line)
    # wall_node = ShockNode(nodef=wall_point)
    # wall_node.set_by_ttab(ttab=dydx2)
    # return wall_node, _i

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
    
    raise NotImplementedError()