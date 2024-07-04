
from aeromoc.moc import MOC2D
from aeromoc.bc import BoundPoints

import numpy as np
import math
from matplotlib import pyplot as plt


if __name__ == '__main__':
    
    n = 15

    kttau = 20.0
    kttal = 12.0
    upperwall = BoundPoints()
    # upperwall.add_section(6 * np.sin(np.linspace(0., math.pi / 180. * ktta, 15)), lambda x: -4. + (6.**2 - x**2)**0.5)
    # upperwall.add_section(np.linspace(0, 5, 12), lambda x: -math.tan(math.pi / 180. * ktta) * x)
    upperwall.add_section(6 * np.sin(np.linspace(0., math.pi / 180. * kttau, 15)), lambda x: 8. - (6.**2 - x**2)**0.5)
    upperwall.add_section(np.linspace(0, 5, 16), lambda x: math.tan(math.pi / 180. * kttau) * x)
    # upperwall.plot()

    lowerwall = BoundPoints()
    lowerwall.add_section(2 * np.sin(np.linspace(0., math.pi / 180. * kttal, 15)), lambda x: -4. + (2.**2 - x**2)**0.5)
    lowerwall.add_section(np.linspace(0, 5, 9), lambda x: -math.tan(math.pi / 180. * kttal) * x)
    
    moc = MOC2D()
    moc.set_boundary('u', typ='wall', points=upperwall)
    moc.set_boundary('l', typ='sym',  y0=0.0)
    # moc.set_boundary('l', typ='wall', points=lowerwall)
    # moc.set_boundary('u', typ='sym',  y0=0.0)

    # init_line = calc_initial_throat_line(n, 2.0, mode='static', p=101325, t=283, mT=2.2)
    moc.calc_initial_line(n, mode='total', p=2015., t=2726., urUp=9., lrUp=3.)

    moc.solve(max_step=100)

    moc.plot_field().show()
    moc.plot_wall(side='u', var=['p', 'ma', 't'], write_to_file='upper.dat')
