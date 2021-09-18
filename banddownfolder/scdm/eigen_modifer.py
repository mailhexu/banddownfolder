import numpy as np

def HR_to_k(HR, Rlist, kpts):

    #Hk[k,:,:] = sum_R (H[R] exp(i2pi k.R))
    phase = np.exp( 2.0j*np.pi *np.tensordot( kpts,Rlist, axes=([1], [1])))
    Hk=np.einsum('rlm, kr -> klm', HR, phase)
    return Hk

def Hk_to_R(Hk, Rlist, kpts, kweights):
    phase = np.exp( -2.0j*np.pi *np.tensordot( kpts,Rlist, axes=([1], [1])))
    HR=np.einsum('klm, kr, k->rlm', Hk, phase, kweights)
    return HR


def modifiy_one_kpoint(evals, evecs, func):
    """
    return the Hamiltonian with same eigen vectors but modified eigen values.
    evals and evecs are the  original eigen values and vectors. 
    return H: the new Hamiltonian
    """
    new_evals=func(evals)
    return evecs @ np.diag(new_evals) @evecs.T.conj()


def ftest(x):
    y=x
    y[-1]+=1
    return y


def test_modify_one_kpoint():
    H0=np.random.random((3,3))
    H0=H0+H0.conj().T
    evals, evecs=np.linalg.eigh(H0)
    print(evecs)

    print("evals: ", evals)
    new_H= modifiy_one_kpoint(evals, evecs, ftest)
    new_evals, new_evecs = np.linalg.eigh(new_H)
    print(new_evals)
    print(new_evecs)



class band_modifier():
    def __init__(self, HR, Rlist, kmesh):
        self.HR=HR
        self.Rlist=Rlist
        self.kmesh=kmesh
        self.kpts=monkhorst_pack(kmesh)


if __name__=="__main__":
    test_modify_one_kpoint()


