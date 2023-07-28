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
from .bc import WallPoints
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
        self.upoints: WallPoints = None
        self.lpoints: WallPoints = None

        self.rrcs: List[List[Node]] = []
        self.lrcs: List[List[Node]] = []

    def set_boundary(self, side: str, typ: str, points: WallPoints = None, y0: int = None) -> None:
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

    def reconstruct_wall(self, dirc):
        _x = []
        _y = []
        _dydx = []

        for cl in (self.lrcs, self.rrcs)[dirc == RIGHTRC]:
            _x.append(cl[0].x)
            _y.append(cl[0].y)
            _dydx.append(cl[0].tta)
            if _dydx[-1] >= BIG_NUMBER:
                print(_x[-1], _y[-1])

        points = WallPoints()
        points.add_section(xx=np.array(_x), yy=np.array(_y), dydx=np.array(_dydx))
        print(points.xx)
        if dirc == RIGHTRC: self.upoints = points
        if dirc == LEFTRC:  self.lpoints = points
                
    def clear(self):
        '''
        clear all the rrcs and lrcs
        '''
        self.rrcs = []
        self.lrcs = []

    def plot_wall(self, side: str, var: str or List[str] = 'p', write_to_file: str = None):
        
        flagw = False
        writes = []

        if isinstance(var, str):
            var = [var]

        if write_to_file is not None:
            f = open(write_to_file, 'w')
            flagw = True

        if flagw: f.write('#moc\nVARIABLES= X')
        
        for v in var:
            if v not in dir(self.rrcs[0][0]):
                print("variable '%s' not in dirctionary" % v)
            
            if flagw: f.write(' %s' % v)

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

            writes.append(ploty)

            plt.plot(plotx, ploty, label=v) 

        plt.legend()
        plt.show()

        if flagw:
            f.write('\nZONE T=MOC, I=%d, F=POINT\n' % len(writes[0]))
            for i in range(len(writes[0])):
                f.write(' %18.9f' % plotx[i])
                for iv in range(len(var)):
                    f.write(' %18.9f' % writes[iv][i])
                f.write('\n')

    def plot_field(self, figure_id=100, write_to_file: str = None, show_figure: bool = False):
        
        plt.figure(figure_id, figsize=(10,4))
        plt.xlim(0, 30)
        plt.ylim(0, 4.5)

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


    def calc_shock_line(self, _lines: List[List[Node]], _xw: float, _yw: float, _dydxw1: float, _dydxw2: float) -> List[ShockNode]:

        # calculate shock node on the initial point (A.)
        wall_point, _i = calc_shock_wall_point(_xw, _yw, _dydxw1, _dydxw2, _lines[-1])
        new_line = [wall_point]

        raise NotImplementedError()


class NOZZLE():
    '''
    main class for nozzle design
    - ideal nozzle for asymmetric nozzle is refer to Mo. 2015 
    '''

    def __init__(self, method: str, pt: float, tt: float, patm: float, asym: float, rup: float, rlow: float) -> None:
        self.convergence = MOC2D()
        self.kernal = MOC2D()
        self.expansion = MOC2D()

        self.method = method

        self.pt = pt
        self.tt = tt
        self.patm = patm
        self.asym = asym    # should be 0~1
        self.r = (rup, rlow)
        self.delta = [0.0, 0.0]

        self.g = 1.4

    @property
    def npr(self) -> float:
        return self.pt / self.patm

    def solve(self):
        
        nthroat = 21
        narc = 15

        # guess initial expansion angle
        main = 1.01
        maexit = (2./(self.g-1) * ((1 + (self.g-1) / 2. * main**2) * self.npr**(1. - 1. / self.g) - 1))**0.5
        # print(maexit)
        deltaU = P_M(self.g, maexit) - P_M(self.g, main)
        pg = 0.
        ttag = 0.
        rup, rlo = self.r

        while abs(pg - self.patm) / self.patm > EPS:
            
            # solve the kernal zone to make pressure on G equal to patm
            deltaL = self.asym * deltaU
            self.delta = [deltaU / DEG, deltaL / DEG]
            # print(self.delta)
            
            upperwall = WallPoints()
            upperwall.add_section(xx=rup * np.sin(np.linspace(0., deltaU, narc)), func=lambda x:  0.5 + rup - (rup**2 - x**2)**0.5)

            lowerwall = WallPoints()
            lowerwall.add_section(xx=rlo * np.sin(np.linspace(0., deltaL, narc)), func=lambda x: -0.5 - rlo + (rlo**2 - x**2)**0.5)
            
            self.kernal.clear()
            self.kernal.set_boundary('u', typ='wall', points=upperwall)
            self.kernal.set_boundary('l', typ='wall', points=lowerwall)
            self.kernal.calc_initial_line(nthroat, mode='total', p=self.pt, t=self.tt, urUp=rup, lrUp=rlo)
            self.kernal.solve(max_step=1000)

            pg = self.kernal.lrcs[-1][-1].p
            ttag = self.kernal.lrcs[-1][-1].tta
            print(pg, ttag / DEG)

            # solve LRCs of C-E to make thetaG = 0.
            dx = rlo * math.sin(deltaL) / 5.
            while abs(self.kernal.lrcs[-1][-1].tta) > EPS: 
                
                lowerwall.add_section(xx=np.array([dx]), func=lambda x: -math.tan(deltaL) * x)
                _xw, _yw, _dydxw = next(lowerwall)
                newlrc = calc_charac_line(_xw, _yw, _dydxw, self.kernal.lrcs[-1], dirc=LEFTRC)

                while newlrc[-1].tta < 0.:

                    lowerwall.del_section(n=1)
                    dx /= 2.
                    lowerwall.add_section(xx=np.array([dx]), func=lambda x: -math.tan(deltaL) * x)
                    _xw, _yw, _dydxw = next(lowerwall)
                    # lowerwall.plot()
                    newlrc = calc_charac_line(_xw, _yw, _dydxw, self.kernal.lrcs[-1], dirc=LEFTRC)
                    # print(lowerwall.xx[-1], dx, newlrc[-1].tta)
                    if SHOWSTEP:
                        plt00 = self.kernal.plot_field()
                        plt00.savefig('%.3f.png'% time.time())
                
                self.kernal.lrcs.append(newlrc)
                self.kernal.rrcs[-1].append(newlrc[-1])

                if SHOWSTEP:
                    plt00 = self.kernal.plot_field()
                    plt00.savefig('%.3f.png'% time.time())
            
            pg = self.kernal.lrcs[-1][-1].p
            ttag = self.kernal.lrcs[-1][-1].tta
            print(pg, self.patm, ttag / DEG)
            # plt00.show()
            deltaU +=  0.2 * (pg / self.patm - 1) * deltaU
        
            # plt100 = self.kernal.plot_field()
            # plt100.show()

        #* solve wall contour
        expansion_dx = self.kernal.lrcs[-1][-1].x - self.kernal.lrcs[-1][-2].x
        old_ll = self.kernal.rrcs[-1]
        self.expansion.rrcs.append(old_ll)  # for plot
        # print(self.kernal.lrcs[-1][-1].lam_plus, self.kernal.lrcs[-1][-2].lam_plus, self.kernal.lrcs[-1][-3].lam_plus)

        while len(old_ll) > 1:
            new_x = expansion_dx + old_ll[-1].x
            new_y = expansion_dx * old_ll[-1].lam_plus + old_ll[-1].y
            newl = calcback_charac_line(new_x, new_y, 0.0, old_ll, RIGHTRC)
            self.expansion.rrcs.append(newl)
            old_ll = newl

            if SHOWSTEP:
                self.expansion.reconstruct_wall(dirc=RIGHTRC)
                plt00 = self.kernal.plot_field()
                plt00 = self.expansion.plot_field()
                plt00.savefig('%.3f.png'% time.time())

        self.expansion.reconstruct_wall(dirc=RIGHTRC)
        plt100 = self.kernal.plot_field()
        plt100 = self.expansion.plot_field().show()

    def cal_conv_section(self, L: float, yt: float, yi: float, nn: float = 50, method: str = 'Witoszynski'):
        '''
        %使用Witoszynski方法计算喷管收缩段形线
        - `L`   (float)     length of the convergence section
        - `yt`  (float)     radius at the throat
        - `yi`  (float)     radius at the inlet
        
        Ref. Witoszynski C. Ueber strahlerweiterung und strahlablablenkung. Vortr ge aus dem gebiete der hydro-und aerodynamik, 1922.
        '''
        x = np.linspace(0., 1., num=nn)

        if method in ['Witoszynski']:
            y = yt * (1. - (1. - (yt / yi)**2) * (((1 - x**2)**2) / ((1 + 1./3. * x**2)**3)))**-0.5
            x_dim = (x - 1.) * L

        self.convergence.upoints = WallPoints()
        self.convergence.upoints.add_section(xx=x_dim, yy=y)
        
    def plot_contour(self, write_to_file: str = None):
        
        for name, color, zone in zip(
            ['Convergence', 'Kernel', 'Expansion'],
            ['r', 'b', 'k'],
            [self.convergence, self.kernal, self.expansion]):

            if zone.upoints is not None:
                plt.plot(zone.upoints.xx, zone.upoints.yy, c=color, label=name)

        plt.show()



