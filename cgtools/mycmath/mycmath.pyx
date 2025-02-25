# mycmath.pyx

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow
from cgtools.utils import timeit, memprofit

@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def calculate_hessian(int resnum,
                      np.ndarray[double, ndim=1] x,
                      np.ndarray[double, ndim=1] y,
                      np.ndarray[double, ndim=1] z,
                      double cutoff=1.2,
                      double spring_constant=1000,
                      int dd=0):
    """
    Calculate the position-position Hessian matrix.
    
    Parameters:
        resnum: Number of residues (atoms)
        x, y, z: 1D NumPy arrays of coordinates (dtype double)
        cutoff: Distance cutoff
        spring_constant: Base spring constant
        dd: Exponent modifier
    Returns:
        A 2D NumPy array (double) of shape (3*resnum, 3*resnum)
    """
    cdef int i, j
    cdef double x_ij, y_ij, z_ij, r, invr, gamma
    cdef np.ndarray[double, ndim=2] hessian = np.zeros((3 * resnum, 3 * resnum), dtype=np.float64)
    
    for i in range(resnum):
        for j in range(resnum):
            if j == i:
                continue
            x_ij = x[i] - x[j]
            y_ij = y[i] - y[j]
            z_ij = z[i] - z[j]
            r = sqrt(x_ij*x_ij + y_ij*y_ij + z_ij*z_ij)
            if r < cutoff:
                invr = 1.0 / r
                gamma = spring_constant * pow(invr, 2 + dd)
            else:
                continue
            # Update diagonal elements (Hii)
            hessian[3 * i, 3 * i]       += gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * i + 1] += gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * i + 2] += gamma * z_ij * z_ij
            hessian[3 * i, 3 * i + 1]     += gamma * x_ij * y_ij
            hessian[3 * i, 3 * i + 2]     += gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * i]     += gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * i + 2] += gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * i]     += gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * i + 1] += gamma * y_ij * z_ij
            # Update off-diagonal elements (Hij)
            hessian[3 * i, 3 * j]       -= gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * j + 1] -= gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * j + 2] -= gamma * z_ij * z_ij
            hessian[3 * i, 3 * j + 1]     -= gamma * x_ij * y_ij
            hessian[3 * i, 3 * j + 2]     -= gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * j]     -= gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * j + 2] -= gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * j]     -= gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * j + 1] -= gamma * y_ij * z_ij
    return hessian


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def hessian(int resnum,
                      np.ndarray[double, ndim=1] x,
                      np.ndarray[double, ndim=1] y,
                      np.ndarray[double, ndim=1] z,
                      double cutoff=1.2,
                      double spring_constant=1000,
                      int dd=0):
    """
    Calculate the position-position Hessian matrix.
    
    Parameters:
        resnum: Number of residues (atoms)
        x, y, z: 1D NumPy arrays of coordinates (dtype double)
        cutoff: Distance cutoff
        spring_constant: Base spring constant
        dd: Exponent modifier
    Returns:
        A 2D NumPy array (double) of shape (3*resnum, 3*resnum)
    """
    cdef int i, j
    cdef double x_ij, y_ij, z_ij, r, invr, gamma
    cdef np.ndarray[double, ndim=2] hessian = np.zeros((3 * resnum, 3 * resnum), dtype=np.float64)
    
    for i in range(resnum):
        for j in range(resnum):
            if j == i:
                continue
            x_ij = x[i] - x[j]
            y_ij = y[i] - y[j]
            z_ij = z[i] - z[j]
            r = sqrt(x_ij*x_ij + y_ij*y_ij + z_ij*z_ij)
            if r < cutoff:
                invr = 1.0 / r
                gamma = spring_constant * pow(invr, 2 + dd)
            else:
                continue
            # Update diagonal elements (Hii)
            hessian[3 * i, 3 * i]       += gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * i + 1] += gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * i + 2] += gamma * z_ij * z_ij
            hessian[3 * i, 3 * i + 1]     += gamma * x_ij * y_ij
            hessian[3 * i, 3 * i + 2]     += gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * i]     += gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * i + 2] += gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * i]     += gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * i + 1] += gamma * y_ij * z_ij
            # Update off-diagonal elements (Hij)
            hessian[3 * i, 3 * j]       -= gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * j + 1] -= gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * j + 2] -= gamma * z_ij * z_ij
            hessian[3 * i, 3 * j + 1]     -= gamma * x_ij * y_ij
            hessian[3 * i, 3 * j + 2]     -= gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * j]     -= gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * j + 2] -= gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * j]     -= gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * j + 1] -= gamma * y_ij * z_ij
    return hessian


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def _perturbation_matrix_old(np.ndarray[double, ndim=2] covariance_matrix,
                             int resnum,
                             double cutoff=1.2,
                             double spring_constant=1000,
                             int dd=0):
    cdef int i, j, k, d, n = resnum
    cdef double norm, sum_val, s
    cdef np.ndarray[double, ndim=2] perturbation_matrix
    cdef np.ndarray[double, ndim=2] directions
    cdef double f0, f1, f2
    cdef double delta0, delta1, delta2

    perturbation_matrix = np.zeros((n, n), dtype=np.float64)
    
    directions = np.array(
        [[1,0,0], [0,1,0], [0,0,1],
         [1,1,0], [1,0,1], [0,1,1], [1,1,1]],
        dtype=np.float64
    )
    for k in range(directions.shape[0]):
        norm = 0.0
        for d in range(3):
            norm += directions[k, d] * directions[k, d]
        norm = sqrt(norm)
        for d in range(3):
            directions[k, d] /= norm

    for k in range(directions.shape[0]):
        f0 = directions[k, 0]
        f1 = directions[k, 1]
        f2 = directions[k, 2]
        for j in range(n):
            for i in range(n):
                delta0 = covariance_matrix[3*i, 3*j] + \
                         covariance_matrix[3*i, 3*j+1] * f1 + \
                         covariance_matrix[3*i, 3*j+2] * f2
                delta1 = covariance_matrix[3*i+1, 3*j] + \
                         covariance_matrix[3*i+1, 3*j+1] * f1 + \
                         covariance_matrix[3*i+1, 3*j+2] * f2
                delta2 = covariance_matrix[3*i+2, 3*j] + \
                         covariance_matrix[3*i+2, 3*j+1] * f1 + \
                         covariance_matrix[3*i+2, 3*j+2] * f2
                s = sqrt(delta0*delta0 + delta1*delta1 + delta2*delta2)
                perturbation_matrix[i, j] += s

    sum_val = 0.0
    for i in range(n):
        for j in range(n):
            sum_val += perturbation_matrix[i, j]
    if sum_val != 0:
        for i in range(n):
            for j in range(n):
                perturbation_matrix[i, j] /= sum_val

    return perturbation_matrix

