import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eigh
from banddownfolder.plot import plot_band
from banddownfolder.scdm.downfolder import downfolding
from banddownfolder.math.linalg import Lowdin
import sisl


class SislWrapper():
    def __init__(self, sisl_hamiltonian):
        self.ham = sisl_hamiltonian

    def solve(self, k):
        return self.ham.eigh(k=k, eigvals_only=False)

    def solve_all(self, kpts, orth=True):
        evals = []
        evecs = []
        for ik, k in enumerate(kpts):
            evalue, evec = self.solve(k)
            if orth:
                S=self.ham.Sk(k, format='dense')
                Smh=Lowdin(S)
                H=self.ham.Hk(k, format='dense')
                Horth=Smh.T.conj()@H@Smh
                evalue, evec = eigh(Horth)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals, dtype=float), np.array(evecs, dtype=complex, order='C')

    def get_fermi_level(self):
        return 0.0


def test():
    fdf = sisl.get_sile('siesta.fdf')
    H = fdf.read_hamiltonian()
    smodel = SislWrapper(H)
    ax=plot_band(model=smodel, npoints=50, erange=[-6,
                                                    8])  # 20 occupied bands
    wann = downfolding(smodel,
                       kmesh=[3, 3, 3],
                       nwann=3,
                       weight_func='window',
                       mu=-6,
                       sigma=0,
                       exclude_bands=[],#list(range(10)),#+list(range(30,67)),
                       use_proj=True,
                       method='scdmk',
                       has_phase=False,
                       selected_basis=[],
                       anchors={(0, 0, 0): tuple(range(14,17))}
                       )
    plot_band(wann, npoints=50, color='green', marker='o', alpha=0.3, ax=ax)
    plt.savefig('Oband.png')
    plt.show()

fdf = sisl.get_sile('siesta.fdf')
H = fdf.read_hamiltonian()

test()
