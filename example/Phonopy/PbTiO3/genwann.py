import numpy as np
from ase.io import write
from banddownfolder.scdm import PhonopyDownfolder
import matplotlib.pyplot as plt

fname = 'phonopy_params.yaml'
downfolder=PhonopyDownfolder(phonopy_yaml=fname)
downfolder.downfold(method='scdmk',nwann=3, #selected_basis=[2,5], 
                    anchors={(.0,.0,.0):(0,1, 2)},
                    use_proj=True, mu=-0.25, sigma=0.4, weight_func='Gauss', kmesh=(3,3,3))
write('POSCAR.vasp', downfolder.model.atoms, vasp5=True)
ax=downfolder.plot_band_fitting(kvectors=np.array([[0. , 0. , 0. ],
           [0.5, 0.0, 0. ],
           [0.5, 0.5, 0.0],
           [0.5 , 0.5 , 0.5 ],
           [0.5, 0.0, 0.0],
           [0.0,0.0,0],
           [0.5,0.5,0.5]                           
                                               ]),
    knames=['$\\Gamma$', 'X','M', 'R', 'X', '$\\Gamma$', "R"], show=False)
plt.savefig('LWF_PTO.pdf')
plt.show()
