import numpy as np
from scipy.linalg import eigh
from banddownfolder.math.linalg import Lowdin


class SislWrapper():
    def __init__(self, sisl_hamiltonian, spin=None):
        self.ham = sisl_hamiltonian
        if self.ham.spin.is_colinear:
            if spin is None:
                raise ValueError("For colinear spin, spin must be given")
        else:
            if spin is not None:
                raise ValueError(
                    "For non-colinear spin and unpolarized spin, spin should be None"
                )
        self.spin = spin

    def solve(self, k):
        if self.spin is None:
            return self.ham.eigh(k=k, eigvals_only=False)
        else:
            return self.ham.eigh(k=k, spin=self.spin, eigvals_only=False)

    def Hk(self, k, format='dense'):
        if self.spin is not None:
            return self.ham.Hk(k, format=format, spin=self.spin)
        else:
            return self.ham.Hk(k, format=format)

    def solve_all(self, kpts, orth=True):
        evals = []
        evecs = []
        for ik, k in enumerate(kpts):
            if orth and self.ham.orthogonal:
                S = self.ham.Sk(k, format='dense')
                Smh = Lowdin(S)
                H = self.Hk(k, format='dense')
                Horth = Smh.T.conj() @ H @ Smh
                evalue, evec = eigh(Horth)
            else:
                evalue, evec = self.solve(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals, dtype=float), np.array(evecs,
                                                      dtype=complex,
                                                      order='C')

    def get_fermi_level(self):
        return 0.0
