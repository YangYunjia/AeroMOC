
import math
import copy
import numpy as np

from typing import Tuple, List, Dict, Callable

from .utils import *
from .node import Node, ShockNode, same_node

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


