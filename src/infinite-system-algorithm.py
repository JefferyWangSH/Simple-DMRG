import time
import numpy as np
from numpy.core.multiarray import datetime_as_string
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
from matplotlib import pyplot as plt


J = 1
Sz = np.array([[0.5, 0], [0, -0.5]], dtype='d')
Splus = np.array([[0, 1], [0, 0]], dtype='d')
model_d = 2

LatticeLength = 100
MaximalStates = 20

class Block:
    len = 1
    dim = model_d
    ham = np.zeros((dim, dim), dtype='d')
    sz = np.array([[0.5, 0], [0, -0.5]], dtype='d')
    splus = np.array([[0, 1], [0, 0]], dtype='d')
 

def enlarge_block():
    # Enlarge each block by a single site.

    # update block hamiltonian
    sysBlock.ham = kron(sysBlock.ham, identity(model_d, dtype='d')) \
                 + J / 2 * ( kron(sysBlock.splus, Splus.conjugate().transpose()) + kron(sysBlock.splus.conjugate().transpose(), Splus) ) \
                 + J * kron(sysBlock.sz, Sz)
    envBlock.ham = kron(identity(model_d, dtype='d'), envBlock.ham) \
                 + J / 2 * ( kron(Splus, envBlock.splus.conjugate().transpose()) + kron(Splus.conjugate().transpose(), envBlock.splus) ) \
                 + J * kron(Sz, envBlock.sz)
    
    # update operator for each block
    sysBlock.splus = kron(identity(sysBlock.dim, dtype='d'), Splus)
    sysBlock.sz = kron(identity(sysBlock.dim, dtype='d'), Sz)
    envBlock.splus = kron(Splus, identity(envBlock.dim, dtype='d'))
    envBlock.sz = kron(Sz, identity(envBlock.dim, dtype='d'))

    sysBlock.dim *= model_d
    envBlock.dim *= model_d
    sysBlock.len += 1
    envBlock.len += 1  

    return sysBlock, envBlock


def form_super_block():
    # Construct the Hamiltonian of super block
    # noninteracting part and interaction between blocks
    superblock_hamiltonian = kron(sysBlock.ham, identity(envBlock.dim, dtype='d')) + kron(identity(sysBlock.dim, dtype='d'), envBlock.ham) \
            + J / 2 * ( kron(sysBlock.splus, envBlock.splus.conjugate().transpose()) + kron(sysBlock.splus.conjugate().transpose(), envBlock.splus) ) \
            + J * kron(sysBlock.sz, envBlock.sz)

    # Diagonalize super hamiltonian
    # Call ARPACK to find the superblock ground state.  ("SA" means find the
    # "smallest in amplitude" eigenvalue.)
    # psi0 corresponds to ground state wavefunction
    (energy,), psi0 = eigsh(superblock_hamiltonian, k=1, which="SA")
    return energy, psi0


def compute_reduced_density_matrix(psi):
    # Construct the reduced density matrix of the system by tracing out the 
    # environment
    # we want to make the (sys, env) indices correspond to (row, column) of a
    # matrix, respectively.
    psi = psi.reshape([sysBlock.dim, envBlock.dim], order="C")
    
    # reduced density matrix \rho = Tr_E |psi><psi|
    sysRho = np.dot(psi, psi.conjugate().transpose())
    envRho = np.dot(psi.conjugate().transpose(), psi)
    return sysRho, envRho


def rotate_and_truncate(rho, block):
    # Diagonalize the reduced density matrix and sort the eigenvectors by
    # eigenvalue.
    evals, evecs = np.linalg.eigh(rho)
    
    possible_eigenstates = []
    for eval, evec in zip(evals, evecs.transpose()):
        possible_eigenstates.append((eval, evec))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `MaximalStates` overall most significant
    # eigenvectors.
    m = min(len(possible_eigenstates), MaximalStates)
    transformation_matrix = np.zeros((block.dim, m), dtype='d', order='F')
    for i, (eval, evec) in enumerate(possible_eigenstates[:m]):
        transformation_matrix[:, i] = evec

    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:m]])

    # Rotate to the low-energy subspace
    block.dim = m
    block.ham = transformation_matrix.conjugate().transpose().dot(block.ham.dot(transformation_matrix))
    block.sz = transformation_matrix.conjugate().transpose().dot(block.sz.dot(transformation_matrix))
    block.splus = transformation_matrix.conjugate().transpose().dot(block.splus.dot(transformation_matrix))
    
    return block, truncation_error


def single_dmrg_step(sysBlock, envBlock):
    """
    Performs a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `MaximalStates` states in the new basis.
    """
    enlarge_block()
    energy, psi0 = form_super_block()
    sysRho, envRho = compute_reduced_density_matrix(psi=psi0)
    sysBlock, sys_truncation_error = rotate_and_truncate(rho=sysRho, block=sysBlock)
    envBlock, env_truncation_error = rotate_and_truncate(rho=envRho, block=envBlock)

    return energy, 0.5 * (sys_truncation_error + env_truncation_error)


def infinite_system_dmrg(sys, env):
    data_list = []
    while (sys.len + env.len) < LatticeLength:
        energy, truncation_error = single_dmrg_step(sys, env)
        data_list.append((sys.len + env.len, energy / (sys.len + env.len), truncation_error))
    return data_list    


if __name__ == "__main__":

    sysBlock, envBlock = Block(), Block()

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
