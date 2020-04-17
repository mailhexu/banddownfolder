from banddownfolder.electron.wannier90 import wannier_to_model
from banddownfolder.scdm.scdmk import WannierProjectedBuilder, WannierScdmkBuilder, occupation_func
from ase.dft.kpoints import monkhorst_pack
from tbmodels import Model
from banddownfolder.electron.ijR import ijR
import matplotlib.pyplot as plt
import pickle
import os
from banddownfolder.plot import plot_band
from banddownfolder.electron.sislwrapper import SislWrapper
import numpy as np


def save_Wannier_model():
    m = Model.from_wannier_folder(folder='/home/hexu/projects/SMO_Wannier/',
                                  prefix='abinito_w90_down')
    mijR = ijR.from_tbmodel(m)
    mijR.save(fname='SMO_tb.nc')


def read_model():
    m = ijR.load_ijR('SMO_tb.nc')
    return m


def downfolding(model,
                kmesh,
                nwann,
                has_phase=False,
                weight_func='unity',
                mu=0,
                sigma=0.01,
                exclude_bands=[],
                use_proj=False,
                method='projected',
                selected_basis=[],
                anchor_kpt=(0,0,0),
                anchors=None):
    k = kmesh[0]
    kpts = monkhorst_pack(kmesh)
    evals, evecs = model.solve_all(kpts)
    try:
        positions = model.positions
    except Exception:
        positions = None
    weight_func = occupation_func(ftype=weight_func, mu=mu, sigma=sigma)
    if method == "scdmk":
        wann_builder = WannierScdmkBuilder(evals=evals,
                                           wfn=evecs,
                                           positions=positions,
                                           kpts=kpts,
                                           nwann=nwann,
                                           weight_func=weight_func,
                                           has_phase=has_phase,
                                           Rgrid=kmesh,
                                           exclude_bands=exclude_bands,
                                           use_proj=use_proj)
        if selected_basis:
            wann_builder.set_selected_cols(selected_basis)
        elif anchors:
            wann_builder.set_anchors(anchors)
        else:
            wann_builder.auto_set_anchors(anchor_kpt)

        wf = wann_builder.get_wannier()
    elif method == "projected":
        wann_builder = WannierProjectedBuilder(
            evals=evals,
            wfn=evecs,
            has_phase=has_phase,
            positions=positions,
            kpts=kpts,
            nwann=nwann,
            weight_func=weight_func,
            Rgrid=kmesh,
            exclude_bands=exclude_bands,
        )
        if selected_basis:
            wann_builder.set_projectors_with_basis(selected_basis)
        elif anchors:
            wann_builder.set_projectors_with_anchors(anchors)
        wf = wann_builder.get_wannier()
    return wf


class BandDownfolder():
    def __init__(self, mijR):
        """
        Setup the model
        """
        self.model = model

    def downfold(self,
                 method='scdmk',
                 kmesh=(5, 5, 5),
                 nwann=0,
                 weight_func='unity',
                 mu=0.0,
                 sigma=2.0,
                 selected_basis=None,
                 anchors=None,
                 anchor_kpt=(0,0,0),
                 use_proj=True,
                 write_hr_nc='Downfolded_hr.nc',
                 write_hr_txt='Downfolded_hr.txt'):
        """
        Downfold the Band structure.
        The method first get the eigenvalues and eigenvectors in a Monkhorst-Pack grid from the model.
        It then use the scdm-k or the projected wannier function method to downfold the Hamiltonian at each k-point.
        And finally it Fourier transform the new basis functions(Wannier functions) from k-space to real space.
        The Hamiltonian can be written.

        Parameters:
        ====================================
        method:  the method of downfolding. scdmk|projected
        kmesh,   The k-mesh used for the BZ sampling. e.g. (5, 5, 5)
                 Note that for the moment, only odd number should be used so that the mesh is Gamma centered.
        nwann,   Number of Wannier functions to be constructed.
        weight_func='Gauss',   # The weight function type. 'unity', 'Gauss', 'Fermi', or 'window'
         - unity: all the bands are equally weighted.
         - Gauss: A gaussian centered at mu, and has the half width of sigma.
         - Fermi: A fermi function. The Fermi energy is mu, and the smearing is sigma.
         - window: A window function in the range of (mu, sigma)
        mu: see above
        sigma=2.0 : see above
        selected_basis, A list of the indexes of the Wannier functions as initial guess. The number should be equal to nwann.
        anchors: Anchor points. The index of band at one k-point. e.g.(0, 0, 0): (6, 7, 8)
        anchor_kpt: the kpoint used for automatically selecting of anchor points.
        use_proj: Whether to use projection to the anchor points in the weight function.
        write_hr_nc: write the Hamiltonian into a netcdf file. It require the NETCDF4 python library. use write_nc=None if not needed.
        write_hr_txt: write the Hamiltonian into a txt file.
        """
        self.ewf = downfolding(self.model,
                               method=method,
                               kmesh=kmesh,
                               nwann=nwann,
                               weight_func=weight_func,
                               mu=mu,
                               sigma=sigma,
                               selected_basis=selected_basis,
                               anchors=anchors,
                               anchor_kpt=anchor_kpt,
                               use_proj=use_proj)
        if write_hr_txt is not None:
            self.ewf.save_txt(write_hr_txt)
        if write_hr_nc is not None:
            self.ewf.write_nc(write_hr_nc)
            return self.ewf

    def plot_band_fitting(self,
                          kvectors=np.array([[0, 0, 0], [0.5, 0, 0],
                                             [0.5, 0.5, 0], [0, 0, 0],
                                             [.5, .5, .5]]),
                          knames=['$\Gamma$', 'X', 'M', '$\Gamma$', 'R'],
                          supercell_matrix=None,
                          npoints=100,
                          efermi=None,
                          erange=None,
                          fullband_color='blue',
                          downfolded_band_color='green',
                          marker='o',
                          ax=None,
                          savefig='Downfolded_band.png',
                          cell=np.eye(3),
                          show=True):
        """
        Parameters:
        ========================================
        kvectors: coordinates of special k-points
        knames: names of special k-points
        supercell_matrix: If the structure is a supercell, the band can be in the primitive cell.
        npoints: number of k-points in the band.
        efermi: Fermi energy.
        erange: range of energy to be shown. e.g. [-5,5]
        fullband_color: the color of the full band structure.
        downfolded_band_color: the color of the downfolded band structure.
        marker: the marker of the downfolded band structure.
        ax: matplotlib axes object.
        savefig: the filename of the figure to be saved.
        show: whether to show the band structure.
        """
        ax = plot_band(self.model,
                       kvectors=kvectors,
                       knames=knames,
                       supercell_matrix=supercell_matrix,
                       npoints=npoints,
                       color=fullband_color,
                       alpha=0.8,
                       marker='',
                       erange=erange,
                       efermi=efermi,
                       cell=cell,
                       ax=ax)
        ax = plot_band(self.ewf,
                       kvectors=kvectors,
                       knames=knames,
                       supercell_matrix=supercell_matrix,
                       npoints=npoints,
                       efermi=efermi,
                       color=downfolded_band_color,
                       alpha=0.5,
                       marker=marker,
                       erange=erange,
                       cell=cell,
                       ax=ax)
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()


class W90Downfolder(BandDownfolder):
    def __init__(self, folder, prefix):
        """
        folder   # The folder containing the Wannier function files
        prefix,   # The prefix of Wannier90 outputs. e.g. wannier90_up
        """

        m = Model.from_wannier_folder(folder=folder, prefix=prefix)
        self.model = ijR.from_tbmodel(m)


class SislDownfolder(BandDownfolder):
    def __init__(self, folder, fdf_file, spin=None):
        """
        Parameters:
        ========================================
        folder: folder of siesta calculation
        fdf_file: siesta input filename
        """
        try:
            import sisl
        except ImportError:
            raise ImportError("sisl is needed. Do you have sisl installed?")
        fdf = sisl.get_sile(os.path.join(folder, fdf_file))
        H = fdf.read_hamiltonian()
        self.model = SislWrapper(H, spin=spin)
