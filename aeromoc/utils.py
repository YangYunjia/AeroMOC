import math
from typing import Tuple, List, Dict, Callable

## Constants

GEOM = 0
GAS_R = 1.4
BIG_NUMBER = 100000.

WALLTYP = ['wall']
SYMTYP  = ['sym']
UPP = ['u', 'upper']
LOW = ['l', 'lower']
LEFTRC = +1
RIGHTRC = -1
PI = math.pi
DEG = math.pi / 180.
EPS = 1e-3
SHOWSTEP = True

## Program define errors

class ExtrapolateError(Exception):
    pass

class SubsonicError(Exception):
    
    def __init__(self, *cors, info) -> None:
        super().__init__('Subsonic flow occurs during "%s" at point (x,y) = %.4f, %.4f' % (info, cors[0], cors[1]))

class KeySelectError(Exception):
    
    def __init__(self, key, value) -> None:
        super().__init__('No value: "%s" for key "%s"' % (value, key))

## thermal functions

def calc_isentropicPTRHO(g: float, ma: float, pTotal: float, tTotal: float) -> Tuple[float, float, float]:
    '''
    Isentropic relation to obtain pressure, temperature, and density from Ma, total pressure, and total temperature
    '''

    ratio = 1 + (g - 1) / 2. * ma**2
    p = pTotal / ratio**(g / (g - 1))
    t = tTotal / ratio
    rho = p / (GAS_R * t)
    return p, t, rho

def P_M(g: float, ma: float) -> float:
    '''
    Prandtl-Meyer expansion angle
    '''

    return ((g+1) / (g-1))**0.5 * math.atan(((g-1) / (g+1)*(ma**2 - 1))**0.5) - math.atan(ma**2 - 1)