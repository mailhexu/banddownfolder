import numpy as np
import copy
from scipy.linalg import eigh
from netCDF4 import Dataset
from ase.dft.kpoints import monkhorst_pack
from ase import Atoms


class LWF():
    """
    Lattice Wannier function
    """
    def __init__(self, wannR, HwannR, Rlist, cell, wann_centers, atoms=None):
        self.atoms = atoms
        self.wannR = wannR
        self.HwannR = HwannR
        self.Rlist = Rlist
        self.cell = cell
        self.wann_centers = wann_centers
        self.nR, self.nbasis, self.nwann = np.shape(wannR)
        self.ndim = np.shape(self.Rlist)[1]
        self.Rdict = {}
        for i, R in enumerate(Rlist):
            self.Rdict[tuple(R)] = i

    @property
    def hoppings(self):
        Rlist = [tuple(R) for R in self.Rlist]
        data = copy.deepcopy(dict(zip(Rlist, self.HwannR)))
        np.fill_diagonal(data[(0, 0, 0)], 0.0)
        return data

    @property
    def site_energies(self):
        iR = self.Rdict[(0, 0, 0)]
        return np.real(np.diag(self.HwannR[iR]))

    def HR(self, R):
        iR = self.Rdict[tuple(R)]
        return self.HwannR[iR]

    def get_wann_Hk(self, k):
        hk = np.zeros((self.nwann, self.nwann), dtype=complex)
        for iR, R in enumerate(self.Rlist):
            phase = np.exp(2j * np.pi * np.dot(R, k))
            hk += self.HwannR[iR, :, :] * phase
        #hk = (hk + hk.T.conj()) / 2
        return hk

    def solve_wann_k(self, k, ham=False):
        hk = self.get_wann_Hk(k)
        evals, evecs = eigh(hk)
        if not ham:
            return evals, evecs
        else:
            return evals, evecs, hk

    def solve_all(self, kpts):
        evals, evecs = [], []
        for k in kpts:
            evalue, evec = self.solve_wann_k(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals), np.array(evecs)

    def get_wann_largest_basis(self):
        ret = []
        for iwann in range(self.nwann):
            a = np.abs(self.wannR[:, :, iwann])
            ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
            ret.append({'R': self.Rlist[ind[0]], 'orb': ind[1]})
        return ret

    def write_nc(self, fname, prefix="wann_", atoms=None):
        root = Dataset(fname, 'w')
        ndim = root.createDimension('ndim', self.ndim)
        nR = root.createDimension('nR', self.nR)
        three = root.createDimension('three', 3)
        nwann = root.createDimension(prefix + 'nwann', self.nwann)
        nbasis = root.createDimension(prefix + 'nbasis', self.nbasis)

        if atoms is not None:
            natom = root.createDimension(prefix + 'natom', len(atoms))

        Rlist = root.createVariable(prefix + 'Rlist',
                                    float,
                                    dimensions=('nR', 'ndim'))
        HamR_real = root.createVariable(prefix + 'HamR_real',
                                       float,
                                       dimensions=('nR', prefix + 'nwann',
                                                   prefix + 'nwann'),
                                       zlib=True)
        HamR_imag = root.createVariable(prefix + 'HamR_imag',
                                       float,
                                       dimensions=('nR', prefix + 'nwann',
                                                   prefix + 'nwann'),
                                       zlib=True)

        wannR_real = root.createVariable(prefix + 'wannier_function_real',
                                         float,
                                         dimensions=('nR', prefix + 'nbasis',
                                                     prefix + 'nwann'),
                                         zlib=True)
        wannR_imag = root.createVariable(prefix + 'wannier_function_imag',
                                         float,
                                         dimensions=('nR', prefix + 'nbasis',
                                                     prefix + 'nwann'),
                                         zlib=True)

        #root.createVariable(prefix+'xred', float64, dimensions=(nR, nwann, nwann))
        #root.createVariable(prefix+'cell', float64, dimensions=(nR, nwann, nwann))
        if atoms is not None:
            cell = root.createVariable(prefix + "cell",
                                       float,
                                       dimensions=('three', 'three'),
                                       zlib=True)
            numbers = root.createVariable(prefix + "atomic_numbers",
                                          int,
                                          dimensions=(prefix + 'natom'),
                                          zlib=True)
            masses = root.createVariable(prefix + "atomic_masses",
                                         float,
                                         dimensions=(prefix + 'natom'),
                                         zlib=True)

            xred = root.createVariable(prefix + "atomic_xred",
                                       float,
                                       dimensions=(prefix + "natom", "three"))

            xcart = root.createVariable(prefix + "atomic_xcart",
                                        float,
                                        dimensions=(prefix + "natom", "three"))
            lwf_masses = root.createVariable(prefix + 'lwf_masses',
                                             float,
                                             dimensions=(prefix + 'nwann'))

            cell[:] = atoms.get_cell().array
            numbers[:] = atoms.get_atomic_numbers()
            xred[:] = atoms.get_scaled_positions()
            xcart[:] = atoms.get_positions()

        Rlist[:] = np.array(self.Rlist)
        HamR_real[:] = np.real(self.HwannR)
        HamR_imag[:] = np.imag(self.HwannR)
        wannR_real[:] = np.real(self.wannR)
        wannR_imag[:] = np.imag(self.wannR)
        root.close()



    def masses_to_lwf_masses(self, masses):
        m3 = np.kron(masses, [1, 1, 1])
        return np.einsum('rij,i->j', (self.wannR.conj()*self.wannR), m3)

    def write_lwf_nc(self, fname, prefix='wann_', atoms=None):
        root = Dataset(fname, 'w')
        ndim = root.createDimension('ndim', self.ndim)
        nR = root.createDimension('nR', self.nR)
        three = root.createDimension('three', 3)
        nwann = root.createDimension(prefix + 'nwann', self.nwann)
        nbasis = root.createDimension(prefix + 'nbasis', self.nbasis)

        if atoms is not None:
            natom = root.createDimension(prefix + 'natom', len(atoms))

        Rlist = root.createVariable(prefix + 'Rlist',
                                    float,
                                    dimensions=('nR', 'ndim'))
        ifc_real = root.createVariable(prefix + 'HamR_real',
                                       float,
                                       dimensions=('nR', prefix + 'nwann',
                                                   prefix + 'nwann'),
                                       zlib=True)
        ifc_imag = root.createVariable(prefix + 'HamR_imag',
                                       float,
                                       dimensions=('nR', prefix + 'nwann',
                                                   prefix + 'nwann'),
                                       zlib=True)

        wannR_real = root.createVariable(prefix + 'wannier_function_real',
                                         float,
                                         dimensions=('nR', prefix + 'nbasis',
                                                     prefix + 'nwann'),
                                         zlib=True)
        wannR_imag = root.createVariable(prefix + 'wannier_function_imag',
                                         float,
                                         dimensions=('nR', prefix + 'nbasis',
                                                     prefix + 'nwann'),
                                         zlib=True)

        wann_centers = root.createVariable(prefix + 'centers',
                                           float,
                                           dimensions=(prefix + 'nwann',
                                                      'three'),
                                           zlib=True)

        #root.createVariable(prefix+'xred', float64, dimensions=(nR, nwann, nwann))
        #root.createVariable(prefix+'cell', float64, dimensions=(nR, nwann, nwann))
        if atoms is not None:
            cell = root.createVariable(prefix + "cell",
                                       float,
                                       dimensions=('three', 'three'),
                                       zlib=True)
            numbers = root.createVariable(prefix + "atomic_numbers",
                                          int,
                                          dimensions=(prefix + 'natom'),
                                          zlib=True)
            masses = root.createVariable(prefix + "atomic_masses",
                                         float,
                                         dimensions=(prefix + 'natom'),
                                         zlib=True)

            xred = root.createVariable(prefix + "atomic_xred",
                                       float,
                                       dimensions=(prefix + "natom", "three"))

            xcart = root.createVariable(prefix + "atomic_xcart",
                                        float,
                                        dimensions=(prefix + "natom", "three"))
            lwf_masses = root.createVariable(prefix + 'lwf_masses',
                                             float,
                                             dimensions=(prefix + 'nwann'))

            cell[:] = atoms.get_cell().array
            numbers[:] = atoms.get_atomic_numbers()
            xred[:] = atoms.get_scaled_positions()
            xcart[:] = atoms.get_positions()
            masses[:] = atoms.get_masses()
            lwf_masses[:] = np.real(self.masses_to_lwf_masses(masses))

        Rlist[:] = np.array(self.Rlist)
        ifc_real[:] = np.real(self.HwannR)
        ifc_imag[:] = np.imag(self.HwannR)
        wannR_real[:] = np.real(self.wannR)
        wannR_imag[:] = np.imag(self.wannR)
        wann_centers[:,:] = self.wann_centers
        root.close()

    @staticmethod
    def load_nc(fname, prefix='wann_'):
        root = Dataset(fname, 'r')
        ndim = root.dimensions['ndim'].size
        nR = root.dimensions['nR'].size
        nwann = root.dimensions[prefix + 'nwann'].size
        nbasis = root.dimensions[prefix + 'nbasis'].size
        try:
            natom = root.dimensions[prefix + 'natom'].size
            has_atom = True
        except:
            has_atom = False
            atoms = None

        Rlist = root.variables[prefix + 'Rlist'][:]
        ifc_real = root.variables[prefix + 'HamR_real'][:]
        ifc_imag = root.variables[prefix + 'HamR_imag'][:]
        # ifc_real = root.variables[prefix + 'ifc_real'][:]
        # ifc_imag = root.variables[prefix + 'ifc_imag'][:]
 
        ifc = ifc_real + ifc_imag * 1.0j

        wannR_real = root.variables[prefix + 'wannier_function_real'][:]
        wannR_imag = root.variables[prefix + 'wannier_function_imag'][:]
        wannR = wannR_real + wannR_imag * 1.0j
        wann_centers = root.variables[prefix + 'centers'][:]
        # FIXME: cell and wann_centers
        if has_atom:
            numbers = root.variables[prefix + 'atomic_numbers'][:]
            xcart = root.variables[prefix + 'xcart'][:]
            cell = root.variables[prefix + 'cell'][:]
            atoms = Atoms(numbers, cell=cell, positions=xcart)
        else:
            atoms = None
        return LWF(wannR,
                   ifc,
                   Rlist,
                   cell=np.eye(3),
                   wann_centers=wann_centers,
                   atoms=atoms)

    def make_supercell(self, sc_maker=None, sc_matrix=None):
        from minimulti.utils.supercell import SupercellMaker
        if sc_maker is None:
            sc_maker = SupercellMaker(sc_matrix)
        if self.atoms is not None:
            sc_atoms = sc_maker.sc_atoms(self.atoms)
        print(self.HwannR.shape)
        sc_Rlist, sc_HR = sc_maker.sc_Rlist_HR(self.Rlist,
                                               self.HwannR,
                                               n_basis=self.nwann)
        return sc_atoms, sc_Rlist, sc_HR

    def get_num_orbitals(self):
        return self.nwann

    def save_txt(self, fname):
        with open(fname, 'w') as myfile:
            myfile.write(f"Number_of_R: {self.nR}\n")
            myfile.write(f"Number_of_Wannier_functions: {self.nwann}\n")
            #myfile.write(f"Cell parameter: {self.cell}\n")
            myfile.write(f"Hamiltonian:  \n" + "=" * 60 + '\n')
            for iR, R in enumerate(self.Rlist):
                myfile.write(f"index of R: {iR}.  R = {R}\n")
                d = self.HwannR[iR]
                for i in range(self.nwann):
                    for j in range(self.nwann):
                        myfile.write(
                            f"R = {R}, i = {i}, j={j} :: H(i,j,R)= {d[i,j]:.4f} \n"
                        )
                myfile.write('-' * 60 + '\n')


class LWF_COHP(LWF):
    def solve_all(self, k_list, eig_vectors=True, total_ham=True):
        nk = len(k_list)
        evals, evecs, hams = np.zeros((self.nwann, nk)), np.zeros(
            (self.nwann, nk, self.nwann), dtype=complex), np.zeros(
                (nk, self.nwann, self.nwann), dtype=complex),
        for ik, k in enumerate(k_list):
            evalue, evec, ham = self.solve_wann_k(k, ham=True)
            evals[:, ik] = evalue
            evecs[:, ik, :] = evec.T
            hams[ik, :, :] = ham
        return evals, evecs, hams

    @staticmethod
    def load_nc(fname, prefix='wann_'):
        root = Dataset(fname, 'r')
        ndim = root.dimensions['ndim'].size
        nR = root.dimensions['nR'].size
        nwann = root.dimensions[prefix + 'nwann'].size
        nbasis = root.dimensions[prefix + 'nbasis'].size

        Rlist = root.variables[prefix + 'Rlist'][:]
        ifc_real = root.variables[prefix + 'ifc_real'][:]
        ifc_imag = root.variables[prefix + 'ifc_imag'][:]
        ifc = ifc_real + ifc_imag * 1.0j

        wannR_real = root.variables[prefix + 'wannier_function_real'][:]
        wannR_imag = root.variables[prefix + 'wannier_function_imag'][:]
        wannR = wannR_real + wannR_imag * 1.0j

        return LWF_COHP(wannR,
                        ifc,
                        Rlist,
                        cell=np.eye(3),
                        wann_centers=np.zeros((nwann, ndim)))


strx = """0.         0.04728673 0.09457345 0.14186018 0.18914691 0.23643363
 0.28372036 0.33100708 0.37829381 0.42558054 0.47286726 0.52015399
 0.56744072 0.61472744 0.66201417 0.7093009  0.75658762 0.80387435
 0.85116108 0.8984478  0.94573453 0.99302125 1.04030798 1.08759471
 1.13488143 1.18216816 1.22945489 1.27674161 1.32402834 1.37131507
 1.41860179 1.46588852 1.51317525 1.56046197 1.6077487  1.65503542
 1.70232215 1.74960888 1.7968956  1.84418233 1.89146906 1.93875578
 1.98604251 2.03332924 2.08061596 2.12790269 2.17518941 2.22247614
 2.26976287 2.31704959 2.36433632 2.41162305 2.45890977 2.5061965
 2.55348323 2.60076995 2.64805668 2.69554905 2.74304143 2.79053381
 2.83802618 2.88551856 2.93301093 2.98050331 3.02799568 3.07548806
 3.12298043 3.17047281 3.21796518 3.26545756 3.31294993 3.36044231
 3.40793469 3.45522141 3.50250814 3.54979486 3.59708159 3.64436832
 3.69165504 3.73894177 3.7862285  3.83351522 3.88080195 3.92808868
 3.9753754  4.02266213 4.06994885 4.11464756 4.15934627 4.20404497
 4.24874368 4.29344239 4.33814109 4.3828398  4.4275385  4.47223721
 4.51693592 4.56163462 4.60633333 4.65103204 4.69573074 4.74042945
 4.78512815 4.82982686 4.87396114 4.91809542 4.96222969 5.00636397
 5.05049825 5.09463253 5.13876681 5.18290108 5.22703536 5.27116964
 5.31530392 5.3594382  5.40357247 5.44770675 5.49184103 5.53653974
 5.58123844 5.62593715 5.67063586 5.71533456 5.76003327 5.80473197
 5.84943068 5.89412939 5.93882809 5.9835268  6.02822551 6.07292421
 6.11762292 6.16232162 6.20702033 6.25171904 6.29309492 6.33447081
 6.37584669 6.41722258 6.45859846 6.49997435 6.54135024 6.58272612
 6.62410201 6.66547789 6.70685378 6.74822966 6.78960555 6.83098144
 6.87235732 6.91373321
"""

strX = """0.         0.66201417 1.32402834 1.98604251 2.64805668 3.40793469
 4.06994885 4.82982686 5.49184103 6.25171904 6.91373321"""

x = tuple(map(float, strx.strip().split()))
X = tuple(map(float, strX.strip().split()))


def test_load():
    kvectors = [[0, 0, 0], [0, 0.5, 0], [.5, .5, 0], [.5, 0, 0], [0, 0, 0],
                [0, 0.25, 0.5], [0.5, 0.25, 0.5], [0.5, 0, 0], [0, 0, 0],
                [0., 0.25, -0.5], [.5, 0.25, -0.5]]
    knames = 'GYCZGBDZGAE'
    fname = "/home/hexu/projects/VO2/results/band/siesta_noU_alpha1.00/Downfolded_hr.nc"
    from ase.io import read
    import matplotlib.pyplot as plt
    from pyDFTutils.ase_utils.kpoints import bandpath
    atoms = read('/home/hexu/projects/VO2/POSCAR_M1.vasp')
    cell = atoms.get_cell()
    band = bandpath(kvectors, cell, npoints=152)
    kpts = band.kpts
    x, X, _labels = band.get_linear_kpoint_axis()

    lwf = LWF.load_nc(fname)
    print(lwf.site_energies)
    lwf = LWF_COHP.load_nc(fname)
    #print(lwf.site_energies)
    Eon = np.real(np.diag(lwf.HR((0, 0, 0))))
    #print(Eon)
    from minimulti.electron.COHP import COHP
    cohp = COHP(lwf)
    ax = cohp.plot_COHP_fatband(
        #iblock=(5, 6, 8, 10),
        #iblock=(4, 5, 7, 9),
        #iblock=(4, 5, 7, 9),
        iblock=(0, 1, 18, 19),
        #jblock=(0, 1, 18, 19),
        #iblock=(0,1,2,3),
        #iblock=(12, 13, 14, 15),
        kpts=kpts,
        k_x=x,
        X=X,
        xnames=knames,
        width=2,
    )
    ax.axhline(linestyle='--', color='black')
    plt.savefig('COHP_alpha1.0.png')
    plt.show()
