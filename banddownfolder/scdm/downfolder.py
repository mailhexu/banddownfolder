import os
import copy
import json
from ase.dft.kpoints import monkhorst_pack
import numpy as np
from banddownfolder.scdm.scdmk import (WannierProjectedBuilder,
                                       WannierScdmkBuilder, occupation_func)
import matplotlib.pyplot as plt
from banddownfolder.plot import plot_band
from banddownfolder.wrapper.ijR import ijR
from banddownfolder.wrapper.wannier90 import wannier_to_model
from banddownfolder.wrapper.sislwrapper import SislWrapper
from banddownfolder.wrapper.phonopywrapper import PhonopyWrapper
from banddownfolder.wrapper.myTB import MyTB


def make_builder(model,
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
                 anchor_kpt=(0, 0, 0),
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
    else:
        raise ValueError("method should be scdmk or projected")
    return wann_builder


class BandDownfolder():
    def __init__(self, model):
        """
        Setup the model
        """
        self.model = model
        self.params = {}

    def set_parameters(self,
                       method='scdmk',
                       kmesh=(5, 5, 5),
                       nwann=0,
                       weight_func='unity',
                       mu=0.0,
                       sigma=2.0,
                       selected_basis=None,
                       anchors=None,
                       anchor_kpt=(0, 0, 0),
                       use_proj=True,
                       exclude_bands=[],
                       post_func=None,
        ):
        """
        Downfold the Band structure.
        The method first get the eigenvalues and eigenvectors in a Monkhorst-Pack grid from the model.
        It then use the scdm-k or the projected wannier function method to downfold the Hamiltonian at each k-point.
        And finally it Fourier transform the new basis functions(Wannier functions) from k-space to real space.
        The Hamiltonian can be written.

        Parameters
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

        self.params = copy.deepcopy(locals())
        self.params.pop('self')
        print(self.params)

    def save_info(self, output_path='./',fname='Downfold.json'):
        results = {'params': self.params}
        with open(os.path.join(output_path, fname), 'w') as myfile:
            json.dump(results, myfile, sort_keys=True, indent=2)


    def downfold(self, post_func=None, 
                       output_path='./',
                       write_hr_nc='Downfolded_hr.nc',
                       write_hr_txt='Downfolded_hr.txt',
                     **params):
        self.params.update(params)
        if 'post_func' in self.params:
            self.params.pop('post_func')
        self.builder = make_builder(self.model, **self.params)
        self.atoms = self.model.atoms
        self.ewf = self.builder.get_wannier()
        if post_func is not None:
            post_func(self.ewf)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            self.save_info(output_path=output_path)
        except:
            pass
        if write_hr_txt is not None:
            self.ewf.save_txt(os.path.join(output_path, write_hr_txt))
        if write_hr_nc is not None:
            self.ewf.write_lwf_nc(os.path.join(output_path, write_hr_nc))
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
        if True:
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
        return ax


class W90Downfolder(BandDownfolder):
    def __init__(self, folder, prefix):
        """
        folder   # The folder containing the Wannier function files
        prefix,   # The prefix of Wannier90 outputs. e.g. wannier90_up
        """
        m = MyTB.read_from_wannier_dir(folder, prefix)
        self.model = m


class SislDownfolder(BandDownfolder):
    def __init__(self,
                 folder=None,
                 fdf_file=None,
                 H=None,
                 spin=None,
                 recover_fermi=False):
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
        shift_fermi = None
        if H is None:
            fdf = sisl.get_sile(os.path.join(folder, fdf_file))
            fdf.read()
            H = fdf.read_hamiltonian()
            self.efermi = fdf.read_fermi_level().data[0]
            if recover_fermi:
                self.shift_fermi = self.efermi
        self.model = SislWrapper(H, spin=spin, shift_fermi=shift_fermi)
        self.atoms = self.model.atoms
        try:
            positions = self.model.positions
        except Exception:
            positions = None
        self.model_info={'orb_names': tuple(self.model.orbs), 
                         'positions': positions.tolist()
            }

    def save_info(self, output_path='./', fname='Downfold.json'):
        cols = self.builder.cols
        self.orbs = [self.model.orbs[i] for i in cols]
        atoms = self.model.atoms
        results = {}
        results['model_info'] = self.model_info
        results['params'] = self.params
        results['results'] = {
            'selected_columns': cols.tolist(),
            'orb_names': tuple(self.orbs),
            'Efermi': self.efermi,
            'chemical_symbols': atoms.get_chemical_symbols(),
            'atom_xred': atoms.get_scaled_positions().tolist(),
            'cell': atoms.get_cell().tolist()
        }
        results.update(self.params)
        with open(os.path.join(output_path, fname), 'w') as myfile:
            json.dump(results, myfile, sort_keys=True, indent=4)

    def wannier_on_grid(self, i, k=None, grid_prec=0.2, grid=None, geom=None):
        '''
        Projects the wannier function on a grid
        '''

        #all_coeffs = DataArray(self.ewf.wannR, dims=('k', 'orb', 'wannier'))
        wannR = self.ewf.wannR

        # Use the geometry of the hamiltonian if the user didn't provide one (usual case)
        if geom is None:
            geom = self.model.ham.geom

        # Create a grid if the user didn't provide one
        if grid is None:
            grid = sisl.Grid(grid_prec, geometry=geom, dtype=complex)

        # Get the coefficients of that we want
        #coeffs = all_coeffs.sel(wannier=i)
        coeffi = wannR[:, :, i]
        #if k is None:
        #    coeffs = coeffs.mean('k')
        #else:
        #    coeffs = coeffs.sel(k=k)

        # Project the orbitals with the coefficients on the grid
        wavefunction(coeffs, grid, geometry=geom)

        return grid


class PhononDownfolder(BandDownfolder):
    def __init__(self, model):
        self.model = model
        try:
            self.atoms = self.model.atoms
        except Exception:
            self.atoms = None
        self.params={}
        


class PhonopyDownfolder(PhononDownfolder):
    def __init__(self, phonon=None, mode='dm', *argv, **kwargs):
        """
        Parameters:
        ========================================
        folder: folder of siesta calculation
        fdf_file: siesta input filename
        """
        try:
            import phonopy
        except ImportError:
            raise ImportError(
                "phonopy is needed. Do you have phonopy installed?")
        if phonon is None:
            phonon = phonopy.load(*argv, **kwargs)
        super().__init__(PhonopyWrapper(phonon, mode=mode))
