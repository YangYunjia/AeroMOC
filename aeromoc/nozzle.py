import math
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Callable

from .utils import *
from .bc import BoundPoints
from .basic import calc_interior_point, calc_charac_line, calc_boundary_point, calc_shock_wall_point, calcback_charac_line, calc_throat_point
from .moc import MOC2D


class NozzleDesign():
    '''
    main class for nozzle design
    '''

    def __init__(self, method: str, 
                 pt: float, tt: float, patm: float, 
                 asym: float = None, rup: float = None, rlow: float = None) -> None:
        '''
        Defines the nozzle design problem

        Currently available:
        - ideal nozzle for asymmetric nozzle (refer to Mo. 2015)

        paras:
        ===

        - `method`: (`str`) method to generate nozzle profile
            Currently available:
            - `idealAsym`: ideal nozzle for asymmetric nozzle (refer to Mo. 2015)

        - nozzle conditions
            - `pt`: (`float`) total inlet pressure (Pa)
            - `tt`: (`float`) total inlet temperature (K)
            - `patm`:   (`float`) ambient pressure (Pa)

        - nozzle contour control parameters
            - `asym`: (`float`) asymmetric parameter 
                >>> asym = deltaL / deltaU
                >>> should be below 1.0
            - `rup`:  (`float`) radius of throat upstream upper contour
            - `rlow`: (`float`) radius of throat upstream lower contour
        
        '''
        
        
        self.convergence = MOC2D()
        self.kernel = MOC2D()
        self.expansion = MOC2D()

        self.method = method

        self.pt = pt
        self.tt = tt
        self.patm = patm
        self.asym = asym    # should be 0~1
        self.r = (rup, rlow)
        self.delta = [0.0, 0.0]

        self.g = GAS_GAMMA

    @property
    def npr(self) -> float:
        return self.pt / self.patm

    def solve(self):

        if self.method in ['idealAsym']:
            self.solve_idealAsym()

    def solve_idealAsym(self):
        '''
        solve for ideal asymmetric nozzle contour
        
        '''
        
        nthroat = 21
        narc = 15

        print('AeroMOC starts to solve for ideal asymmetric nozzle')
        print('-------------\n')

        print('AeroMOC guess the initial expansion angle')
        # guess initial expansion angle
        main = 1.01
        maexit = (2./(self.g-1) * ((1 + (self.g-1) / 2. * main**2) * self.npr**(1. - 1. / self.g) - 1))**0.5
        # print(maexit)
        deltaU = P_M(self.g, maexit) - P_M(self.g, main)
        pg = 0.
        ttag = 0.
        rup, rlo = self.r
        print(f'Guess initial expansion angle = {deltaU / DEG:.3f}')
        print('-------------\n')

        print('AeroMOC starts to iterate the kernel region')

        while abs(pg - self.patm) / self.patm > EPS:
            
            # solve the kernal zone to make pressure on G equal to patm
            deltaL = self.asym * deltaU
            self.delta = [deltaU / DEG, deltaL / DEG]
            # print(self.delta)
            
            upperwall = BoundPoints()
            upperwall.add_section(xx=rup * np.sin(np.linspace(0., deltaU, narc)), func=lambda x:  0.5 + rup - (rup**2 - x**2)**0.5)

            lowerwall = BoundPoints()
            lowerwall.add_section(xx=rlo * np.sin(np.linspace(0., deltaL, narc)), func=lambda x: -0.5 - rlo + (rlo**2 - x**2)**0.5)
            
            self.kernel._clear()
            self.kernel.set_boundary('u', typ='wall', points=upperwall)
            self.kernel.set_boundary('l', typ='wall', points=lowerwall)
            self.kernel.calc_initial_line(nthroat, mode='total', p=self.pt, t=self.tt, urUp=rup, lrUp=rlo)
            self.kernel.solve(max_step=1000)

            pg = self.kernel.lrcs[-1][-1].p
            ttag = self.kernel.lrcs[-1][-1].tta

            # solve LRCs of C-E to make thetaG = 0.
            dx = rlo * math.sin(deltaL) / 5.
            while abs(self.kernel.lrcs[-1][-1].tta) > EPS: 
                
                lowerwall.add_section(xx=np.array([dx]), func=lambda x: -math.tan(deltaL) * x)
                _xw, _yw, _dydxw = next(lowerwall)
                newlrc = calc_charac_line(_xw, _yw, _dydxw, self.kernel.lrcs[-1], dirc=LEFTRC)

                while newlrc[-1].tta < 0.:

                    lowerwall.del_section(n=1)
                    dx /= 2.
                    lowerwall.add_section(xx=np.array([dx]), func=lambda x: -math.tan(deltaL) * x)
                    _xw, _yw, _dydxw = next(lowerwall)
                    # lowerwall.plot()
                    newlrc = calc_charac_line(_xw, _yw, _dydxw, self.kernel.lrcs[-1], dirc=LEFTRC)
                    # print(lowerwall.xx[-1], dx, newlrc[-1].tta)
                    if SHOWSTEP:
                        plt00 = self.kernel.plot_field()
                        plt00.savefig('%.3f.png'% time.time())
                
                self.kernel.lrcs.append(newlrc)
                self.kernel.rrcs[-1].append(newlrc[-1])

                if SHOWSTEP:
                    plt00 = self.kernel.plot_field()
                    plt00.savefig('%.3f.png'% time.time())
            
            pg = self.kernel.lrcs[-1][-1].p
            ttag = self.kernel.lrcs[-1][-1].tta
            # print(pg, self.patm, ttag / DEG)
            # plt00.show()
            print(f'>   deltaU = {deltaU / DEG:.2f}, pg = {pg:.2f}, thetag = {ttag / DEG:.2f}')

            deltaU +=  0.2 * (pg / self.patm - 1) * deltaU
        
            # plt100 = self.kernal.plot_field()
            # plt100.show()

        print('kernel region succueesfully solved')
        print('-------------\n')

        print('AeroMOC starts to solve the wall contour')
        #* solve wall contour
        expansion_dx = self.kernel.lrcs[-1][-1].x - self.kernel.lrcs[-1][-2].x
        old_ll = self.kernel.rrcs[-1]
        self.expansion.rrcs.append(old_ll)  # for plot
        # print(self.kernal.lrcs[-1][-1].lam_plus, self.kernal.lrcs[-1][-2].lam_plus, self.kernal.lrcs[-1][-3].lam_plus)

        while len(old_ll) > 1:
            new_x = expansion_dx + old_ll[-1].x
            new_y = expansion_dx * old_ll[-1].lam_plus + old_ll[-1].y
            newl = calcback_charac_line(new_x, new_y, 0.0, old_ll, RIGHTRC)
            self.expansion.rrcs.append(newl)
            old_ll = newl

            if SHOWSTEP:
                self.expansion._reconstruct_wall(dirc=RIGHTRC)
                plt00 = self.kernel.plot_field()
                plt00 = self.expansion.plot_field()
                plt00.savefig('%.3f.png'% time.time())

        print('Nozzle solving complete')
        self.expansion._reconstruct_wall(dirc=RIGHTRC)
        plt100 = self.kernel.plot_field()
        plt100 = self.expansion.plot_field().show()

    def gen_conv_section(self, L: float, yt: float, yi: float, nn: float = 50, method: str = 'Witoszynski'):
        '''
        generate nozzle contour of the convergence section
        - `L`   (float)     length of the convergence section
        - `yt`  (float)     radius at the throat
        - `yi`  (float)     radius at the inlet
        - `nn`  (int)       number of the points
        - `method` (str)    method to calculate the contour, default is the Witoszynski method(1)
        
        (1) Ref. Witoszynski C. Ueber strahlerweiterung und strahlablablenkung. Vortr ge aus dem gebiete der hydro-und aerodynamik, 1922.
        '''
        x = np.linspace(0., 1., num=nn)

        if method in ['Witoszynski']:
            y = yt * (1. - (1. - (yt / yi)**2) * (((1 - x**2)**2) / ((1 + 1./3. * x**2)**3)))**-0.5
            x_dim = (x - 1.) * L

        self.convergence.upoints = BoundPoints()
        self.convergence.upoints.add_section(xx=x_dim, yy=y)

        if self.method in ['idealAsym']:
            self.convergence.lpoints = BoundPoints()
            self.convergence.lpoints.add_section(xx=x_dim, yy=-y)            
    
    def write_contour(self, file_name: str = 'contour', var: List[str] = ['p']):

        # self.convergence.write_contour(file_name, [], 'CONVERGENCE', write_headlines=True)
        self.kernel.write_contour_values(file_name + '_values.dat', var, 'KERNEL', write_headlines=True)
        self.expansion.write_contour_values(file_name + '_values.dat', var, 'EXPANSION', write_headlines=False)

        subfix = '_geometry.dat'
        self.convergence.write_contour_geometry(file_name + subfix, 'CONVERGENCE', write_headlines=True)
        self.kernel.write_contour_geometry(file_name + subfix, 'KERNEL', write_headlines=False)
        self.expansion.write_contour_geometry(file_name + subfix, 'EXPANSION', write_headlines=False)

    def plot_contour(self, write_to_file: str = None):

        plt.figure(figsize=(10, 4))        
        for name, color, zone in zip(
            ['Convergence', 'Kernel', 'Expansion'],
            ['r', 'b', 'k'],
            [self.convergence, self.kernel, self.expansion]):

            plt.plot(zone.upoints.xx, zone.upoints.yy, c=color, label=name)
            if zone.upoints_blc is not None:
                plt.plot(zone.upoints_blc.xx, zone.upoints_blc.yy, '--', c=color, label=name+'_blc')
            
        if self.method in ['idealAsym']:
            for color, zone in zip(
                                ['r', 'b'],
                                [self.convergence, self.kernel]):
                plt.plot(zone.lpoints.xx, zone.lpoints.yy, c=color)
                if zone.lpoints_blc is not None:
                    plt.plot(zone.lpoints_blc.xx, zone.lpoints_blc.yy, '--', c=color)
        plt.legend()
        plt.show()

    def apply_blc(self):

        # decide the starting point of the boundary layer
        xi = self.kernel.upoints.xx[-1]
        yi = self.kernel.upoints.yy[-1]
        dydxi = self.kernel.upoints.dydx[-1]
        x0 = xi - yi / dydxi

        # calculate the boundary layer correction with the edenfield method
        for zone in [self.kernel, self.expansion]:
            zone.boundary_layer_correction(x0, 'edenfield', 'design', t0=self.tt, tw=None)
        
        # shift the convergence section
        self.convergence.upoints_blc = BoundPoints()
        self.convergence.lpoints_blc = BoundPoints()
        xx_bls = self.convergence.upoints.xx
        yy_bls = self.convergence.upoints.yy + (self.kernel.upoints_blc.yy[0] - self.convergence.upoints.yy[-1]) * (xx_bls - xx_bls[0]) / (xx_bls[-1] - xx_bls[0])
        self.convergence.upoints_blc.add_section(xx=xx_bls, yy=yy_bls)
        self.convergence.lpoints_blc.add_section(xx=xx_bls, yy=-yy_bls)
