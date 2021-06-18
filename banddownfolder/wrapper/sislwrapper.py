import numpy as np
from scipy.linalg import eigh
from banddownfolder.math.linalg import Lowdin
from ase.atoms import Atoms
from banddownfolder.utils.symbol import symbol_number
from collections import defaultdict


class SislWrapper():
    def __init__(self, sisl_hamiltonian, shift_fermi=None, spin=None):
        self.ham = sisl_hamiltonian
        self.shift_fermi=shift_fermi
        self.spin=spin
        self.orbs=[]
        self.orb_dict=defaultdict(lambda:[])
        g=self.ham._geometry
        _atoms=self.ham._geometry._atoms
        atomic_numbers=[]
        atom_positions=g.xyz
        self.cell=np.array(g.sc.cell)
        for ia, a in enumerate(_atoms):
            atomic_numbers.append(a.Z)
        self.atoms=Atoms(numbers=atomic_numbers, cell=self.cell, positions=atom_positions)
        xred=self.atoms.get_scaled_positions()
        sdict=list(symbol_number(self.atoms).keys())
        if self.ham.spin.is_colinear:
            if spin is None:
                raise ValueError("For colinear spin, spin must be given")
        else:
            if spin is not None:
                raise ValueError(
                    "For non-colinear spin and unpolarized spin, spin should be None"
                )

        self.positions=[]
        if self.ham.spin.is_colinear:
            for ia, a in enumerate(_atoms):
                symnum=sdict[ia]
                orb_names=[]
                for x in a.orbitals:
                    name=f"{symnum}|{x.name()}|{spin}"
                    orb_names.append(name)
                    self.positions.append(xred[ia])
                self.orbs+=orb_names
                self.orb_dict[ia]+=orb_names
            self.norb = len(self.orbs)
            self.nbasis=self.norb
        elif self.ham.spin.is_spinorbit:
            for spin in ['up', 'down']:
                for ia, a in enumerate(_atoms):
                    symnum=sdict[ia]
                    orb_names=[]
                    for x in a.orbitals:
                        name=f"{symnum}|{x.name()}|{spin}"
                        orb_names.append(name)
                        self.positions.append(xred[ia])
                    self.orbs+=orb_names
                    self.orb_dict[ia]+=orb_names
            self.norb=len(self.orbs)/2
            self.nbasis= len(self.orbs)
        else:
            for ia, a in enumerate(_atoms):
                symnum=sdict[ia]
                orb_names=[]
                for x in a.orbitals:
                    name=f"{symnum}|{x.name()}|None"
                    orb_names.append(name)
                    self.positions.append(xred[ia])
                self.orbs+=orb_names
                self.orb_dict[ia]+=orb_names
            self.norb=len(self.orbs)
            self.nbasis= len(self.orbs)
        self.positions=np.array(self.positions, dtype=float)


    def print_orbs(self):
        print(self.orb_dict)

    def solve(self, k):
        if self.spin is None:
            evals, evecs= self.ham.eigh(k=k, eigvals_only=False)
        else:
            evals, evecs= self.ham.eigh(k=k, spin=self.spin, eigvals_only=False)
        if self.shift_fermi:
            evals += self.shift_fermi
        return evals, evecs

    def Hk(self, k, format='dense'):
        if self.spin is not None:
            return self.ham.Hk(k, format=format, spin=self.spin)
        else:
            return self.ham.Hk(k, format=format)

    def solve_all(self, kpts, orth=False):
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
        if self.shift_fermi:
            return self.shift_fermi
        else:
            return 0.0


