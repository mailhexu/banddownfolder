import numpy as np
from scipy.linalg import eigh
from ase.dft.kpoints import monkhorst_pack
from netCDF4 import Dataset


class LWF():
    """
    Lattice Wannier function
    """
    def __init__(self, wannR, HwannR, Rlist):
        self.wannR = wannR
        self.HwannR = HwannR
        self.Rlist = Rlist
        self.nR, self.nbasis, self.nwann = np.shape(wannR)
        self.ndim = np.shape(self.Rlist)[1]

    def get_wann_Hk(self, k):
        hk = np.zeros((self.nwann, self.nwann), dtype=complex)
        for iR, R in enumerate(self.Rlist):
            phase = np.exp(2j * np.pi * np.dot(R, k))
            hk += self.HwannR[iR, :, :] * phase
        #hk = (hk + hk.T.conj()) / 2
        return hk

    def solve_wann_k(self, k):
        hk = self.get_wann_Hk(k)
        return eigh(hk)

    def solve_all(self, kpts):
        evals, evecs = [], []
        for k in kpts:
            evalue, evec = self.solve_wann_k(k)
            evals.append(evalue)
            evecs.append(evec)
        return np.array(evals), np.array(evecs)

    def write_nc(self, fname, prefix='wann_'):
        root = Dataset(fname, 'w')
        ndim = root.createDimension('ndim', self.ndim)
        nR = root.createDimension('nR', self.nR)
        three = root.createDimension('three', self.nR)
        nwann = root.createDimension(prefix + 'nwann', self.nwann)
        nbasis = root.createDimension(prefix + 'nbasis', self.nbasis)

        Rlist = root.createVariable(prefix + 'Rlist',
                                    float,
                                    dimensions=('nR', 'ndim'))
        ifc_real = root.createVariable(prefix + 'ifc_real',
                                       float,
                                       dimensions=('nR',prefix+ 'nwann', prefix+'nwann'))
        ifc_imag = root.createVariable(prefix + 'ifc_imag',
                                       float,
                                       dimensions=('nR', prefix+'nwann', prefix+'nwann'))

        wannR_real = root.createVariable(prefix + 'wannier_function_real',
                                         float,
                                         dimensions=('nR', prefix+'nbasis', prefix+'nwann'))
        wannR_imag = root.createVariable(prefix + 'wannier_function_imag',
                                         float,
                                         dimensions=('nR', prefix+'nbasis', prefix+'nwann'))

        #root.createVariable(prefix+'xred', float64, dimensions=(nR, nwann, nwann))
        #root.createVariable(prefix+'cell', float64, dimensions=(nR, nwann, nwann))
        Rlist[:] = np.array(self.Rlist)
        ifc_real[:] = np.real(self.HwannR)
        ifc_imag[:] = np.imag(self.HwannR)
        wannR_real[:] = np.real(self.wannR)
        wannR_imag[:] = np.imag(self.wannR)
        root.close()

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
