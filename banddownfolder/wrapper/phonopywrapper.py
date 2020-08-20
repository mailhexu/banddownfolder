from scipy.linalg import eigh
import numpy as np
from phonopy import load, Phonopy
from ase import Atoms


class PhonopyWrapper():
    def __init__(self, phonon: Phonopy, mode='dm'):
        self.phonon = phonon
        self._prepare()
        prim = self.phonon.get_primitive()
        self.atoms = Atoms(prim.get_chemical_symbols(),
                           positions=prim.get_positions(),
                           cell=prim.get_cell())
        self._positions = np.repeat(self.atoms.get_scaled_positions(),
                                    3,
                                    axis=0)
        self.dr = self._positions[None, :, :] - self._positions[:, None, :]
        self.mode=mode.lower()
        assert self.mode in ['ifc', 'dm']
        if self.mode=='ifc':
            masses=np.kron(self.atoms.get_masses(), [1,1,1])
            self.Mmat=np.sqrt(masses[:, None]*masses[None, :])

    def _prepare(self):
        self.phonon.symmetrize_force_constants()
        self.phonon.symmetrize_force_constants_by_space_group()

    def solve(self, k):
        #self.phonon._set_dynamical_matrix()
        #if self.phonon._dynamical_matrix is None:
        #    msg = ("Dynamical matrix has not yet built.")
        #    raise RuntimeError(msg)

        Hk = self.phonon.get_dynamical_matrix_at_q(k)
        Hk *= np.exp(-2.0j * np.pi * np.einsum('ijk, k->ij', self.dr, k))
        if self.mode=='ifc':
            Hk*=self.Mmat
        evals, evecs = eigh(Hk)
        return evals, evecs

    def solve_all(self, kpts):
        evals = []
        evecs = []
        for ik, k in enumerate(kpts):
            evalue, evec = self.solve(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals, dtype=float), np.array(evecs,
                                                      dtype=complex,
                                                      order='C')

    @property
    def positions(self):
        return self._positions


def test():
    fname = '/home/hexu/projects/VO2/gen/phonopy_params.yaml'
    phonon = load(phonopy_yaml=fname)
    phon = PhonopyWrapper(phonon)
    eva, eve = phon.solve([0.5, 0.5, 0.5])
    print(eva, eve)
    #print(phon.get_positions())


if __name__ == '__main__':
    test()
