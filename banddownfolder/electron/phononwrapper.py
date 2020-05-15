from scipy.linalg import eigh

class PhonopyWrapper():
    def __init__(self, phonon):
        self.phonon=phonon
        self._dynamical_matrix=self.phonon._set_dynamical_matrix()
        if self._dynamical_matrix is None:
            msg = ("Dynamical matrix has not yet built.")
            raise RuntimeError(msg)

    def solve(self, k):
        self._dynamical_matrix.run(k)
        dm = self._dynamical_matrix.get_dynamical_matrix()
        evals, evecs = eigh(dm)
        return evals, evecs

    def solve_all(self, kpts):
        evals=[]
        evecs=[]
        for ik, k in enumerate(kpts):
            evalue, evec = self.solve(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals, dtype=float), np.array(evecs,
                                                      dtype=complex,
                                                      order='C')

    def get_positions(self):
        pass
