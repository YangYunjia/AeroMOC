
from aeromoc.moc import NOZZLE

if __name__ == '__main__':

    nz = NOZZLE('ideal', pt=200000, tt=1393, patm=5000, asym=0.7, rup=3., rlow=3.)
    nz.solve()