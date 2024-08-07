'''
This is the main file of the pyMOC(AeroMOC) program. The program is initially developed
by Yunjia Yang (yyj980401@126.com). It is also a option of the assignment of Prof. Yufei 
Zhang's (zhangyufei@tsinghua.edu.cn) curriculum "Advanced Aerodynamics".

The program is open to further development. Please follow the instruction to fork and raise
pull request.

Yunjia Yang, Apr. 13, 2023

'''
import math
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Callable

from .utils import *
from .node import Node, ShockNode
from .bc import BoundPoints, blc_edenfield, blc_linear_estimate
from .basic import calc_interior_point, calc_charac_line, calc_boundary_point, calc_shock_wall_point, calcback_charac_line, calc_throat_point


class MOC2D():
    '''
    This function is to calculate a supersonic flowfield use Method of Characteristic.

    The solution zone is assigned by two boundary wall (upper and lower). The supersonic
    flow enter the zone from left side (init_line), and propagate rightward.

    The solution process is DIMENSIONAL.
    '''

    def __init__(self) -> None:
        self.utyp: str = None
        self.ltyp: str = None
        self.uy0 = 1.0
        self.ly0 = 0.0
        self.upoints: BoundPoints = None
        self.lpoints: BoundPoints = None
        self.upoints_blc: BoundPoints = None
        self.lpoints_blc: BoundPoints = None
        self.udata: np.ndarray = None
        self.ldata: np.ndarray = None

        self.rrcs: List[List[Node]] = []
        self.lrcs: List[List[Node]] = []

    def set_boundary(self, side: str, typ: str, points: BoundPoints = None, y0: int = None) -> None:
        '''
        This function is used to set the upper or lower boundary wall. The wall is depicted
        by the class `WallPoints`.

        ### para:
        - side (= `UPP` or `LOW`) upper or lower side
        - typ  (= `WALL` or `SYM`)
        - points (`WallPoints`) a generator for wall points
        - rUp  (float) upward radius (for nozzle's init_line)
        
        '''
        if side in UPP:
            self.utyp = typ         # upper boundary type (wall or sym)
            self.upoints = points   # the `WallPoint` class of upper 
            self.uy0  = points[0][1]       

        elif side in LOW:
            self.ltyp = typ
            self.lpoints = points
            if typ == 'wall':
                self.ly0  = points[0][1]
            else:
                self.ly0 = y0

    def _dimen(self, ip: Node, dirc: int) -> Node:
        '''
        This function is used to dimensionalize points from initial line calculaiton.

        Remark that it is ONLY used to above propose.

        ### para:
        - ip:   a point given by initial line calculation
        - dirc: direction of characteristic line

        ### return:
        new dimensionalized point
        '''
        
        newp = copy.deepcopy(ip)
        newp.x = (self.uy0 - self.ly0) * ip.x
        if dirc == RIGHTRC:
            newp.y = self.ly0 + (self.uy0 - self.ly0) * ip.y
        else:
            newp.y = self.uy0 - (self.uy0 - self.ly0) * ip.y
            newp.tta = -ip.tta

        return newp

    def calc_initial_line(self, n: int, mode: str = 'total', 
                                 p: float = 0.0, t: float = 0.0, mT: float = 0.0,
                                 urUp: float = 0.0, lrUp: float = 0.0,
                                 **para: Dict):
        '''
        This function is used to generate init_line.

        Initialization is by calculating a c.l. from the first point of each wall boundary. The flow variable on 
        every points on the initial c.l is decided by the one of following two ways:
        - `total`   Use NASA's method to calculate the mach number and theta of the points according to the 
                    distance between the points and the wall, given the total pressure (`p`), temperature (`t`), 
                    and upstream wall radius (`urUp` and `lrUp`). (see reference) 
        - `static`  Set the mach number and theta by a given number (`mT`), theta is set to 0
        - `profile` **TODO**: Set the mach number and theta on the initial line by a profile

        After the mach number and theta being decided, the static pressure and temperature can be calculated by
        isotropic relations. Then the angle of c.l. from this point can be obtained, and we can step to next point
        on the initial line.

        If the upper and lower boundary conditions are both `wall`, then two initial line is origined from both sides.
        After each iteration, whether two lines are intersecting is checked.

        ### para:
        - `n` (int)   amount of points on the initial line
        - `mode` (str)  way to initialize. 
        - `p`, `t`      total pressure and temperature
        - `mT`          mach number
        - `urUp`, `lrUp`    upstream radius
        - `LineMaMax`, `LineMaMin`  The maximum and minmum mach number on the initial line. If the mach number calculated
                                    from the `total` method exceed the range, a sub-iteration is called to change the new 
                                    point's position to avoid it(by moving the new point's `x`). According to NASA, the max
                                    mach number is suggested to not exceed 1.5.
        - `InitPointsDistri`    (= `equal` or `sin`) The distribution of the initial points on the `y` direction.
        
        ### reference:
        interpreted from CalcInitialThroatLine <- MOC_GidCalc_BDE <- the MOC program of NASA
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
                _yy = np.sin(PI * (n - np.arange(n + 1)) / n / 2.)**pows
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
                npoint = calc_throat_point(_yy[ir], nrline, mode, urUp, p, t, mT, lmmax, lmmin)
                # npoint is non-dimension, nrline is non-dimensional line used to calculate next 
                # point, the real point is stored in rline
                nrline.append(npoint)
                rline.append(self._dimen(npoint, RIGHTRC))
                
                # record initial line
                if len(rline) > 1:
                    rline[-1].lastnode(rline[-2], RIGHTRC)

                # determine if left and right initial line intersect
                if len(lline) >= (n-(ir-1)+1) and (rline[ir-1].x - lline[n-(ir-1)].x) * (rline[ir].x - lline[n-ir].x) <= 0.0:
                    cpoint = calc_interior_point(rline[ir-1], lline[n-ir])
                    lline = lline[:(n-ir)+1] + [cpoint]
                    rline = rline[:(ir-1)+1] + [cpoint]
                    break
                
                ir += 1

            if self.ltyp in WALLTYP:
                npoint = calc_throat_point(1.-_yy[il], nlline, mode, lrUp, p, t, mT, lmmax, lmmin)
                nlline.append(npoint)
                lline.append(self._dimen(npoint, LEFTRC))
                
                # record initial line
                if len(rline) > 1:
                    lline[-1].lastnode(lline[-2], LEFTRC)
                
                # determine if left and right initial line intersect
                if len(rline) >= ((il+1)+1) and (rline[il].x - lline[n-il].x) * (rline[il+1].x - lline[n-(il+1)].x) <= 0.0:
                    cpoint = calc_interior_point(rline[il], lline[n-(il+1)])
                    lline = lline[:(n-(il+1))+1] + [cpoint]
                    rline = rline[:il        +1] + [cpoint]
                    break

                il -= 1

        self.lrcs.append(lline)
        self.rrcs.append(rline)

    def solve(self, max_step: int = 100000):
        '''
        Solve the flowfield with MOC

        ## para:

        - `max_step`: maximum steps to calculate
        
        '''
        ## check boundary and initial condition
        if self.ltyp is None or self.utyp is None:
            raise RuntimeError("Types for upper or lower boundary conditions are not set\n Use `set_boundary`")
        if len(self.rrcs) < 1 or len(self.lrcs) < 1:
            raise RuntimeError("Not initialized\n Use `calc_initial_line`")

        t1 = time.time()
        step = 0

        while step < max_step:

            newlrc: List[Node] = []
            newrrc: List[Node] = []

            # upper wall
            if self.utyp in WALLTYP:
                try:
                    lastrrc = self.rrcs[-1]
                    _xw, _yw, _dydxw = next(self.upoints)
                    newrrc = calc_charac_line(_xw, _yw, _dydxw, lastrrc, dirc=RIGHTRC)
                    if self.ltyp in SYMTYP:
                        newrrc.append(calc_boundary_point('free', -RIGHTRC, p2=newrrc[-1], p3=lastrrc[-1])[0])
                except StopIteration:
                    pass
            # lower wall
            if self.ltyp in WALLTYP:
                try:
                    lastlrc = self.lrcs[-1]
                    _xw, _yw, _dydxw = next(self.lpoints)
                    newlrc = calc_charac_line(_xw, _yw, _dydxw, lastlrc, dirc=LEFTRC)
                    if self.utyp in SYMTYP:
                        # print(len(newrrc), len(lastrrc))
                        newlrc.append(calc_boundary_point('free', -LEFTRC, p2=newlrc[-1], p3=lastlrc[-1])[0])
                except StopIteration:
                    pass
            # point on the center line
            if len(newlrc) > 0 and len(newrrc) > 0:
                cpoint = calc_interior_point(newrrc[-1], newlrc[-1])

                newlrc.append(cpoint)
                newrrc.append(cpoint)
                self.lrcs.append(newlrc)
                self.rrcs.append(newrrc)
            
            # if only left or right c.l. exist, add the last point as the end of the other side's c.l.
            elif len(newlrc) > 0 and len(newrrc) <= 0:
                self.lrcs.append(newlrc)
                self.rrcs[-1].append(newlrc[-1])
            elif len(newrrc) > 0 and len(newlrc) <= 0:
                self.rrcs.append(newrrc)
                self.lrcs[-1].append(newrrc[-1])
            else:
                break
            step += 1

            if SHOWSTEP:
                plt00 = self.plot_field()
                plt00.savefig('%.3f.png'% time.time())

        if SHOWSTEP:
            plt00.show()
        
        t2 = time.time()
        print('Solve done in %.3f s' % (t2 - t1))

    def _reconstruct_wall(self, dirc):
        '''
        reconstruct the wall with the starting point of the characteristic lines

        only used when reconstructing expansion contour when generate nozzle
        
        '''
        _x = []
        _y = []
        _dydx = []

        for cl in (self.lrcs, self.rrcs)[dirc == RIGHTRC]:
            _x.append(cl[0].x)
            _y.append(cl[0].y)
            _dydx.append(cl[0].tta)
            if _dydx[-1] >= BIG_NUMBER:
                print(_x[-1], _y[-1])

        points = BoundPoints()
        points.add_section(xx=np.array(_x), yy=np.array(_y), dydx=np.array(_dydx))
        if dirc == RIGHTRC: self.upoints = points; self.utyp = 'wall'
        if dirc == LEFTRC:  self.lpoints = points; self.ltyp = 'wall'
                
    def _clear(self):
        '''
        clear all the rrcs and lrcs
        '''
        self.rrcs = []
        self.lrcs = []

    def get_boundary_variables(self):
        '''
        variables: x, y, rho, p, vel, theta, ma, t
                   0, 1,   2, 3,   4,     5,  6, 7 
        '''

        for idx, rcs in enumerate([self.rrcs, self.lrcs]):
            data = []
            for cl in rcs:  
                if len(cl) > 0:
                    data.append([cl[0].x, cl[0].y, cl[0].rho, cl[0].p, cl[0].vel, cl[0].tta, cl[0].ma, cl[0].p / (cl[0].rho * GAS_R)])
            if idx == 0:
                self.udata = np.array(data).transpose()
            else:
                self.ldata = np.array(data).transpose()
        
    def write_contour_values(self, file_name: str = 'contour.dat', var:List[str] = ['p'],
                      zone_name: str = '', write_headlines: bool = False):
        
        if write_headlines:
            f = open(file_name, 'w')
            f.write('# moc contour values generated by AeroMOC\n')
            f.write('VARIABLES = X Y')
            for v in var:
                if v not in dir(self.rrcs[0][0]):
                    print("variable '%s' not in dirctionary" % v)
                    var.remove(v)
                else:
                    f.write(' %s' % v)
            f.write('\n')

        else:
            f = open(file_name, 'a')
        
        for side, charlns in zip(['UPPER', 'LOWER'], [self.rrcs, self.lrcs]):
            writes = []
            for l in charlns:
                if len(l) > 0:
                    temp = [l[0].x, l[0].y]
                    for v in var:
                        temp.append(l[0].__getattribute__(v))
                    writes.append(temp)

            f.write(f'\nZONE T="{side}_{zone_name}", I={len(writes):d}, F=POINT\n')
            for ln in writes:
                for lnvar in ln:
                    f.write(' %18.9f' % lnvar)
                f.write('\n')

        f.close()

    def write_contour_geometry(self, file_name: str = 'contour.dat', zone_name: str = '', write_headlines: bool = False):
        
        if write_headlines:
            f = open(file_name, 'w')
            f.write('# moc contour geometry generated by AeroMOC\n')
            f.write('VARIABLES = X Y')
            f.write('\n')

        else:
            f = open(file_name, 'a')

        zone_sub_names = ['UPPER', 'LOWER', 'UPPERBLC', 'LOWERBLC']
        zone_sub_objs  = [self.upoints, self.lpoints, self.upoints_blc, self.lpoints_blc]

        not_none_indexs = [i for i, x in enumerate(zone_sub_objs) if x is not None]
        zone_sub_names  = [zone_sub_names[i] for i in not_none_indexs]
        zone_sub_objs   = [zone_sub_objs[i] for i in not_none_indexs]
        
        for side, charlns in zip(zone_sub_names, zone_sub_objs):
            length = len(charlns.xx)
            f.write(f'\nZONE T="{side}_{zone_name}", I={length:d}, F=POINT\n')
            for i in range(length):
                f.write(' %18.9f %18.9f\n' % (charlns.xx[i], charlns.yy[i]))

        f.close()

    def plot_wall(self, side: str, var: List[str] = ['p']):
        
        for v in var:
            if v not in dir(self.rrcs[0][0]):
                print("variable '%s' not in dirctionary" % v)
            
            plotx = []
            ploty = []

            if side in UPP: 
                for l in self.rrcs:
                    if len(l) > 0:
                        plotx.append(l[0].x)
                        ploty.append(l[0].__getattribute__(v))

            if side in LOW: 
                for l in self.lrcs:
                    if len(l) > 0:
                        plotx.append(l[0].x)
                        ploty.append(l[0].__getattribute__(v))

            plt.plot(plotx, ploty, label=v) 

        plt.legend()
        plt.show()

    def plot_field(self, figure_id=100, write_to_file: str = None, show_figure: bool = False):
        
        plt.figure(figure_id, figsize=(10,4))
        # plt.xlim(0, 30)
        # plt.ylim(0, 4.5)

        for line in self.lrcs + self.rrcs:
            for nd in line:
                if nd.streamnode is not None:
                    plt.plot([nd.streamnode[0], nd.x], [nd.streamnode[1], nd.y], '-', c='k')
                if nd.leftnode is not None:
                    plt.plot([nd.leftnode[0], nd.x], [nd.leftnode[1], nd.y], '-', c='r')
                if nd.rightnode is not None:
                    plt.plot([nd.rightnode[0], nd.x], [nd.rightnode[1], nd.y], '-', c='b')
        
        self.upoints.plot()
        
        if write_to_file is not None:
            plt.savefig(write_to_file)
        if show_figure:
            plt.show()
        
        return plt

    def boundary_layer_correction(self, x0, method='edenfield', mode='design', **kwargs):

        self.get_boundary_variables()
        for idx, data in enumerate([self.udata, self.ldata]):

            if len(data) == 0: continue

            if method == 'edenfield':
                dys = blc_edenfield(xx=data[0], x0=x0, pp=data[3], ma=data[6], tt=data[7], t0=kwargs['t0'], tw=kwargs['tw'])
            elif method == 'linear':
                dys = blc_linear_estimate(xx=data[0], x0=x0, Me=data[6][-1])

            dys = np.maximum(0.0, dys)

            if mode == 'design':

                if idx == 0 and self.utyp in WALLTYP:
                    self.upoints_blc = BoundPoints()
                    self.upoints_blc.add_section(data[0] - dys * np.sin(data[5]), data[1] + dys * np.cos(data[5]))
                if idx == 1 and self.ltyp in WALLTYP:
                    self.lpoints_blc = BoundPoints()
                    self.lpoints_blc.add_section(data[0] - dys * np.sin(data[5]), data[1] - dys * np.cos(data[5]))

            elif mode == 'simulation':
                self.upoints_blc = copy.deepcopy(self.upoints)
                xx_blc = data[0] + dys * np.sin(data[5])
                yy_blc = data[1] - dys * np.cos(data[5])
                self.upoints = BoundPoints()
                self.upoints.add_section(xx_blc, yy_blc)            

    def calc_shock_line(self, _lines: List[List[Node]], _xw: float, _yw: float, _dydxw1: float, _dydxw2: float) -> List[ShockNode]:

        # calculate shock node on the initial point (A.)
        wall_point, _i = calc_shock_wall_point(_xw, _yw, _dydxw1, _dydxw2, _lines[-1])
        new_line = [wall_point]

        raise NotImplementedError()

