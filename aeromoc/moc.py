'''
MOC main function

YangYunjia, 2023.1.3

'''
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Callable

from .utils import *
from .node import Node, ShockNode, same_node
from .bc import WallPoints
from .basic import calc_interior_point, calc_charac_line, calc_sym_point, calc_shock_wall_point
from .nozzle import calc_throat_point


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
                    cpoint, _ = calc_interior_point(rline[il], lline[n-(il+1)])
                    lline = lline[:(n-(il+1))+1] + [cpoint]
                    rline = rline[:il        +1] + [cpoint]
                    break

                il -= 1

        plt.plot([ip.x for ip in rline], [ip.y for ip in rline], '-x', c='b')
        plt.plot([ip.x for ip in lline], [ip.y for ip in lline], '-x', c='r')

        self.lrcs.append(lline)
        self.rrcs.append(rline)

    def solve(self, max_step: int = 100000):
        
        step = 0

        while step < max_step:

            # upper wall
            if self.utyp in WALLTYP:
                try:
                    lastrrc = self.rrcs[-1]
                    _xw, _yw, _dydxw = next(self.upoints)
                    newrrc = calc_charac_line(_xw, _yw, _dydxw, lastrrc, dirc=RIGHTRC)
                    if self.ltyp in SYMTYP:
                        newrrc.append(calc_sym_point(newrrc[-1], lastrrc[-1], dirc=RIGHTRC))
                except EndofwallError:
                    pass
            # lower wall
            if self.ltyp in WALLTYP:
                try:
                    lastlrc = self.lrcs[-1]
                    _xw, _yw, _dydxw = next(self.lpoints)
                    newlrc = calc_charac_line(_xw, _yw, _dydxw, lastlrc, dirc=LEFTRC)
                    if self.utyp in SYMTYP:
                        # print(len(newrrc), len(lastrrc))
                        newlrc.append(calc_sym_point(newlrc[-1], lastlrc[-1], dirc=LEFTRC))
                except EndofwallError:
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
        
        plt.show()

    def clear(self):
        self.rrcs = []
        self.lrcs = []

    def plot_wall(self, side: str, var: str or List[str] = 'p', wtf: str = None):
        
        flagw = False
        writes = []

        if isinstance(var, str):
            var = [var]

        if wtf is not None:
            f = open(wtf, 'w')
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
        self.kernal = MOC2D()
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
        
        nthroat = 11
        narc = 15

        # guess initial expansion angle
        main = 1.01
        maexit = (2./(self.g-1) * ((1 + (self.g-1) / 2. * main**2) * self.npr**(1. - 1. / self.g) - 1))**0.5
        print(maexit)
        deltaU = P_M(self.g, maexit) - P_M(self.g, main)
        pg = 0.
        ttag = 0.
        rup, rlo = self.r

        while abs(pg - self.patm) / self.patm > EPS:
            
            # solve the kernal zone
            deltaL = self.asym * deltaU
            print(pg, self.patm, ttag / DEG)
            self.delta = [deltaU / DEG, deltaL / DEG]
            print(self.delta)
            
            upperwall = WallPoints()
            upperwall.add_section(rup * np.sin(np.linspace(0., deltaU, narc)), lambda x:  0.5 + rup - (rup**2 - x**2)**0.5)

            lowerwall = WallPoints()
            lowerwall.add_section(rlo * np.sin(np.linspace(0., deltaL, narc)), lambda x: -0.5 - rlo + (rlo**2 - x**2)**0.5)
            
            self.kernal.clear()
            self.kernal.set_boundary('u', typ='wall', y0= 0.5, points=upperwall, rUp=rup)
            self.kernal.set_boundary('l', typ='wall', y0=-0.5, points=lowerwall, rUp=rlo)
            self.kernal.calc_initial_throat_line(nthroat, mode='total', p=self.pt, t=self.tt)
            self.kernal.solve(max_step=500)

            # solve LRCs of C-E to make thetaG = 0.
            dx = rlo * math.sin(deltaL) / 5.

            while True:
                lowerwall.add_section(np.array([dx]), lambda x: -math.tan(deltaL) * x)
                _xw, _yw, _dydxw = next(lowerwall)
                newlrc = calc_charac_line(_xw, _yw, _dydxw, self.kernal.lrcs[-1], dirc=LEFTRC)
                
                if abs(newlrc[-1].tta) < EPS:
                    # only add the last lrc to kernal
                    self.kernal.lrcs.append(newlrc)
                    self.kernal.rrcs[-1].append(newlrc[-1])
                    break     

                while newlrc[-1].tta < 0.:

                    lowerwall.del_section(n=1)
                    dx /= 2.
                    lowerwall.add_section(np.array([dx]), lambda x: -math.tan(deltaL) * x)
                    _xw, _yw, _dydxw = next(lowerwall)
                    # lowerwall.plot()
                    newlrc = calc_charac_line(_xw, _yw, _dydxw, self.kernal.lrcs[-1], dirc=LEFTRC)
                    print(lowerwall.xx[-1], dx, newlrc[-1].tta)
                
                self.kernal.lrcs.append(newlrc)
                self.kernal.rrcs[-1].append(newlrc[-1])

            pg = self.kernal.lrcs[-1][0].p
            ttag = self.kernal.lrcs[-1][0].tta
            deltaU +=  0.2 * (pg / self.patm - 1) * deltaU

            # 




