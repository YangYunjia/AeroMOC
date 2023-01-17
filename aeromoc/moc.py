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
from .basic import calc_interior_point, calc_wall_point, calc_sym_point, calc_shock_wall_point
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
        
        plt.show()

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
