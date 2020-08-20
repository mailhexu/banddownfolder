



## Lattice Wannier functions



### Build supercell from LWF

The LWF is defined in a supercell, with the cell index $\vec{R}$. Then for each atom $i$ in cell $\vec{R}$, the displacement is $\vec{W}_h( R, i)$ for the wannier function $h$ . When we need to build a larger supercell of the original cell, the cell index is $\vec{T}$. The amplitude of the displacements  is thus a $\vec{D}(\vec{T}^{\prime}, i)=\sum_h f(h, \vec{T}) \vec{W}_h(\vec{R}, i)\delta({\vec{T}^\prime, \vec{T}+\vec{R}})$

The displacement 





### LWF-Hubbard model

 $$H=\sum_{ij}A_{ij}Q_iQ_j  +  \sum_iA_{i4}Q_i^4 + \sum_{ij}A_{i2j2}Q_i^2Q_j^2 $$

where $Q$ is the amplitude of the lattice wannier functions built from the two unstable phonon branches.  distrubuted in a supercell. The  $A$ values are the interaction coefficients. The $A_{ij}$ is from the harmonic phonons and are exact. The other $A$ parameters are fitted. Here I restrict the $A_{i2j2}$ to only the on-site interaction (between two lattice wannier functions in the same cell), which is effectively a competition between the two distortions. 

Then we simulate the temperature dependent structure  with this model by a Monte Carlo method. At low temperature, it is at the M1 phase, where all the Q's have the equivalent amplitude. As temperature increases and domain walls emerges.  Eventually the structure becomes disordered and has an average Rutile structure.



By controlling the competition between the two modes, we could tune the tendency to form M2 structures. The M2-R transition is simulated with another set of parameters with larger $A_{i2j2}$. 