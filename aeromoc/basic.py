import math
import copy
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Callable

from .utils import *
from .node import Node, ShockNode, same_node


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

def _calc_p3_from_last_line(last_line: List[Node], llidx0: int, didx: int, p4x: float, p4y: float, p4tta: float) -> Tuple[Node, int]:

    p3 = Node()

    llidx = llidx0   # node index on the last rrc line

    while llidx >= 0 and llidx < len(last_line) - 1:
    
        _p50 = last_line[llidx]
        _p51 = last_line[llidx + 1]
        ratio = _calc_p3_xy(_p50.x, _p50.y, _p51.x, _p51.y, p4x, p4y, p4tta)
        if ratio < 1.0 and ratio >= 0.0:
            break

        llidx += didx

    else:
        raise ExtrapolateError()

    p3.cors = _p51.cors + ratio * (_p50.cors - _p51.cors)
    p3.vals = _p51.vals + ratio * (_p50.vals - _p51.vals)

    return p3, llidx

def _calc_p4_xy(p1x: float, p1y: float, p1tta: float, p2x: float, p2y: float, p2tta: float) -> Tuple[float, float]:
    p4x = ((p1y - p2y) - (p1tta * p1x - p2tta * p2x)) / (p2tta - p1tta)
    p4y = p2y + p2tta * (p4x - p2x)
    return p4x, p4y

def _calc_p4_vals(_p1: Node, _p2: Node, _p4: Node, last_line: List[Node] = None, llidx: int = 0) -> int:

    T_plus  = _p2.Q * _p2.p + _p2.tta - _p2.S_plus  * (_p4.x - _p2.x)
    T_minus = _p1.Q * _p1.p - _p1.tta - _p1.S_minus * (_p4.x - _p1.x)
    _p4.p = (T_plus + T_minus) / (_p1.Q + _p2.Q)
    _p4.tta = T_plus - _p2.Q * _p4.p

    if last_line is None:
        # the point 3 is obtained from p.1 and p.2
        ratio = _calc_p3_xy(_p1.x, _p1.y, _p2.x, _p2.y, _p4.x, _p4.y, _p4.tta)        # the ratio of point 3 on the line p1-p2 from p2
        _p3 = Node()
        _p3.vals = _p2.vals + ratio * (_p1.vals - _p2.vals)
    else:
        _p3 = _calc_p3_from_last_line(last_line, llidx, -1, _p4.x, _p4.y, _p4.tta)

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

def calc_interior_point(p1: Node, p2: Node, last_line: List[Node] = None, llidx: int = 0) -> Node or Tuple[Node, int]:
    '''
    calculate interior point location and values

    paras:
    ===
    `p1`, `p2`:    the origins of the c.l, all values should be prescribed
        it doesn't matter p4 get from p1, p2 by forward or backward

    `last_line`:    (optional) when p4 is get from backward, the location of p4
    is get from p1 and p2 by c.l, the rho, vel is get from find a p3 on the last
    c.l. When `last_line` is prescribed, the code automatically see p4 as backward
    and to find the p3 on the prescribed last c.l.

    `llidx`:    (optional, default is 0) the idx on the last line to start search with

    returns:
    ==
    (`p4`, `llidx`)
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

    plt.plot([p1.x, p4.x], [p1.y, p4.y], '-', c='b')
    plt.plot([p2.x, p4.x], [p2.y, p4.y], '-', c='r')

    if last_line is None:
        return p4
    else:
        return p4, _llidx


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
        # when can't find a interaction point between the lrc to the wall point, and the last rrc
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

def calc_exitplane_point(xx: float, yy: float, extta: float, last_line: List[Node or ShockNode], dirc: int) -> Tuple[Node, int]:

    p4  =  Node(xx, yy)
    p4.tta = extta  # exit theta
    _p2 = copy.deepcopy(last_line[-1])

    p3, llidx = _calc_p3_from_last_line(last_line, len(last_line) - 2, -1, xx, yy, extta)
    _p3 = copy.deepcopy(p3)

    if dirc == LEFTRC:
        _calc_boundary_p4_vals(_p3, p4, _p2=_p2)
    else:
        _calc_boundary_p4_vals(_p3, p4, _p1=_p2)
    _p2.vals = 0.5 * (_p2.vals + p4.vals)
    _p3.vals = 0.5 * (_p3.vals + p4.vals)
    if dirc == LEFTRC:
        _calc_boundary_p4_vals(_p3, p4, _p2=_p2)
    else:
        _calc_boundary_p4_vals(_p3, p4, _p1=_p2)

    plt.plot([_p3.x, p4.x], [_p3.y, p4.y], '--', c='k')
    plt.plot([_p2.x, p4.x], [_p2.y, p4.y], '-', c='r')

    return p4, llidx

def calcback_charac_line(_xe: float, _ye: float, _extta: float, lastcl: List[Node], dirc:int):

    newcl: List[Node] = []
    exit_point, llidx = calc_exitplane_point(_xe, _ye, _extta, lastcl, dirc)
    newcl.insert(0, exit_point)

    try:
        for _ii in range(len(lastcl[-1], 0, -1)):
            if dirc == RIGHTRC:
                newp, llidx = calc_interior_point(newcl[0], lastcl[_ii-1], lastcl, llidx)
            else:
                raise NotImplementedError()
    except ExtrapolateError:
        # point 3 is out of last c.l, which means point 4 beyond wall
        raise NotImplementedError()




def calc_charac_line(_xw: float, _yw: float, _dydxw: float, lastcl: List[Node], dirc: int) -> List[Node]:

    newcl: List[Node] = []
    wall_point, _i = calc_wall_point(_xw, _yw, _dydxw, lastcl, dirc=dirc)
    newcl.append(wall_point)
    
    # calculate interior points amount = (N_lastline - 1)
    for _ii in range(_i + 1, len(lastcl)):
        if dirc == RIGHTRC:
            newcl.append(calc_interior_point(newcl[-1], lastcl[_ii]))
        else:
            newcl.append(calc_interior_point(lastcl[_ii], newcl[-1]))

    return newcl

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