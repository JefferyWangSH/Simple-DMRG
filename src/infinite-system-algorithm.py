import time
import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
from matplotlib import pyplot as plt

'''
Simple infinite dmrg codes for simulations of 1d xxz Heisenberg chain.
The program is implemented by conventional dmrg algorithm, with no fabrications of MPS states.

TODO:
    1. Measurements of observable quantities.
    2. Matirx product state (MPS) implementation. 
'''

J = 1
Sz = np.array([[0.5, 0], [0, -0.5]], dtype='d')
Splus = np.array([[0, 1], [0, 0]], dtype='d')
model_d = 2


class Block:
    'Block structure for dmrg simulation'

    def __init__(self, single_site_dim=2):
        self.len = 1
        self.dim = single_site_dim
        self.ham = np.zeros((self.dim, self.dim), dtype='d')
        self.density_matrix = np.zeros((self.dim, self.dim), dtype='d')
        self.sz = np.array([[0.5, 0], [0, -0.5]], dtype='d')
        self.splus = np.array([[0, 1], [0, 0]], dtype='d')


    def enlarge_block(self, enlarge_dirt):
        '''
        Enlarge each block by a single site.
        In our convention, we enlarge the 'self' block from right in case of enlarge_dirt = 1,
        and from left when enlarge_dirt = -1.
        '''
        assert(abs(enlarge_dirt) == 1.0)

        # update block hamiltonian and operators.
        if enlarge_dirt == +1:
            self.ham = kron(self.ham, identity(model_d, dtype='d')) \
                     + 0.5 * J * ( kron(self.splus, Splus.conjugate().transpose()) + kron(self.splus.conjugate().transpose(), Splus) ) \
                     + J * kron(self.sz, Sz)
            self.splus = kron(identity(self.dim, dtype='d'), Splus)
            self.sz = kron(identity(self.dim, dtype='d'), Sz)
            self.dim *= model_d
            self.len += 1

        if enlarge_dirt == -1:
            self.ham = kron(identity(model_d, dtype='d'), self.ham) \
                     + 0.5 * J * ( kron(Splus, self.splus.conjugate().transpose()) + kron(Splus.conjugate().transpose(), self.splus) ) \
                     + J * kron(Sz, self.sz)
            self.splus = kron(Splus, identity(self.dim, dtype='d'))
            self.sz = kron(Sz, identity(self.dim, dtype='d'))
            self.dim *= model_d
            self.len += 1


    def form_super_block(self, block):
        '''
        This member function constructs the Hamiltonian of super block,
        and diagonalize to obtain ground state wavefunction |psi0> and energy e0.
        Mind that the superblock is formed in such a way that characterized by |self> * |block>
        '''
        assert(isinstance(block, Block))

        # Construct the Hamiltonian of super block
        # including noninteracting part and interaction between blocks
        superblock_hamiltonian = kron(self.ham, identity(block.dim, dtype='d')) + kron(identity(self.dim, dtype='d'), block.ham) \
            + 0.5 * J * ( kron(self.splus, block.splus.conjugate().transpose()) + kron(self.splus.conjugate().transpose(), block.splus) ) \
            + J * kron(self.sz, block.sz)
        
        # Diagonalize super hamiltonian
        # Call ARPACK to find the superblock ground state.  ("SA" means find the
        # "smallest in amplitude" eigenvalue.)
        # psi0 corresponds to ground state wavefunction
        (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")
        return energy, psi0


    def rotate_and_truncate(self):
        # Diagonalize the reduced density matrix and sort the eigenvectors by
        # eigenvalue.
        evals, evecs = np.linalg.eigh(self.density_matrix)
    
        possible_eigenstates = []
        for eval, evec in zip(evals, evecs.transpose()):
            possible_eigenstates.append((eval, evec))
        possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

        # Build the transformation matrix from the `MaximalStates` overall most significant
        # eigenvectors.
        m = min(len(possible_eigenstates), MaximalStates)
        transformation_matrix = np.zeros((self.dim, m), dtype='d', order='F')
        for i, (eval, evec) in enumerate(possible_eigenstates[:m]):
            transformation_matrix[:, i] = evec

        truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:m]])

        # Rotate to the low-energy subspace
        self.dim = m
        self.ham = transformation_matrix.conjugate().transpose().dot(self.ham.dot(transformation_matrix))
        self.sz = transformation_matrix.conjugate().transpose().dot(self.sz.dot(transformation_matrix))
        self.splus = transformation_matrix.conjugate().transpose().dot(self.splus.dot(transformation_matrix))
    
        return truncation_error



def compute_reduced_density_matrix(sys, env, psi):
    # Construct the reduced density matrix of the system by tracing out the 
    # environment
    # we want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.
    assert(isinstance(sys, Block))
    assert(isinstance(env, Block))

    psi = psi.reshape([sys.dim, env.dim], order="C")
    
    # reduced density matrix \rho = Tr_E |psi><psi|
    sys.density_matrix = np.dot(psi, psi.conjugate().transpose())
    env.density_matrix = np.dot(psi.conjugate().transpose(), psi)


def single_dmrg_step(sys, env):
    """
    Performs a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `MaximalStates` states in the new basis.
    """
    assert(isinstance(sys, Block))
    assert(isinstance(env, Block))

    sys.enlarge_block(+1)
    env.enlarge_block(-1)
    energy, psi0 = sys.form_super_block(env)
    compute_reduced_density_matrix(sys, env, psi0)
    sys_truncation_error = sys.rotate_and_truncate()
    env_truncation_error = env.rotate_and_truncate()

    return energy, 0.5 * (sys_truncation_error + env_truncation_error)


def infinite_system_dmrg(sys, env):
    '''
    Subroutine to perform one signle infinite dmrg simulation 
    '''
    assert(isinstance(sys, Block))
    assert(isinstance(env, Block))

    data_list = []
    while (sys.len + env.len) < LatticeLength:
        energy, truncation_error = single_dmrg_step(sys, env)
        data_list.append((sys.len + env.len, energy / (sys.len + env.len), truncation_error))
    return data_list    


if __name__ == "__main__":

    LatticeLength = 100
    MaximalStates = 20

    sysBlock = Block(single_site_dim=model_d)
    envBlock = Block(single_site_dim=model_d)

    start_time = time.time()
    data = infinite_system_dmrg(sys=sysBlock, env=envBlock)
    end_time = time.time()
    
    print("Time cost: {:.3f} s".format(end_time - start_time))

    length, energy_per_site, truncation_error = zip(*data)
    
    plt.figure()
    plt.grid(linestyle='-.')
    plt.errorbar(length, energy_per_site, truncation_error,
        label="${\\beta = 4.0}$", ms=4, fmt='o:', ecolor='r', elinewidth=1.5, capsize=4)
    plt.xlabel("${L}$")
    plt.ylabel("${E/L}$")
    plt.show()
    