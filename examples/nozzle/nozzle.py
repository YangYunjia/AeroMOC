
from aeromoc.moc import NOZZLE

if __name__ == '__main__':

    nz = NOZZLE('idealAsym', pt=200000, tt=1393, patm=5000, asym=0.7, rup=3., rlow=3.)
    nz.cal_conv_section(L=2., yt=0.5, yi=1.)
    nz.solve()
    nz.apply_blc()
    nz.plot_contour()