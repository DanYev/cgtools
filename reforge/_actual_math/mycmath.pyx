# mycmath.pyx

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, pow
from reforge.utils import timeit, memprofit

@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def _calculate_hessian(int resnum,
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
    
    double *array = (double*)calloc(n * n * sizeof(double));
    if (array == NULL) {
        // handle allocation error
    }
    // To set all values to zero:
    for (int i = 0; i < n * n; i++) {
        array[i] = 0.0;
    }
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
def _hessian(np.ndarray[double, ndim=2] vec,
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
        A 2D NumPy array (double) 
    """
    cdef int i, j
    cdef double x_ij, y_ij, z_ij, r, invr, gamma
    cdef int n = vec.shape[0] // 3
    cdef np.ndarray[double, ndim=2] hessian 
    
    hessian = np.zeros((3 * n, 3 * n), dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            x_ij = vec[i, 0] - vec[j, 0]
            y_ij = vec[i, 1] - vec[j, 1]
            z_ij = vec[i, 2] - vec[j, 2]
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
                             int resnum):
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


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def _perturbation_matrix(np.ndarray[double, ndim=2] covariance_matrix) -> np.ndarray:
    """
    Compute a perturbation matrix from a covariance matrix.
    
    Parameters
    ----------
    covariance_matrix : ndarray (shape = (3*m, 3*n))
        A covariance matrix computed from positions.
        
    Returns
    -------
    perturbation_matrix : ndarray (shape = (m, n))
        A perturbation matrix, normalized by its total sum.
    """
    cdef int i, j, k, d
    cdef int m = covariance_matrix.shape[0] // 3
    cdef int n = covariance_matrix.shape[1] // 3
    cdef double norm, sum_val, s
    cdef np.ndarray[double, ndim=2] perturbation_matrix = np.zeros((m, n), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] directions
    cdef double f0, f1, f2
    cdef double delta0, delta1, delta2

    # Create an array of 7 direction vectors (7 x 3) and normalize each row.
    directions = np.array([[1,0,0], [0,1,0], [0,0,1],
                           [1,1,0], [1,0,1], [0,1,1], [1,1,1]],
                          dtype=np.float64)
    for k in range(directions.shape[0]):
        norm = 0.0
        for d in range(3):
            norm += directions[k, d] * directions[k, d]
        norm = sqrt(norm)
        for d in range(3):
            directions[k, d] /= norm

    # Loop over each direction vector.
    for k in range(directions.shape[0]):
        f0 = directions[k, 0]
        f1 = directions[k, 1]
        f2 = directions[k, 2]
        for j in range(n):
            for i in range(m):
                # Compute dot product of the 3x3 block and the direction vector.
                # Block is covariance_matrix[3*i:3*i+3, 3*j:3*j+3].
                delta0 = (covariance_matrix[3*i,   3*j]   * f0 +
                          covariance_matrix[3*i,   3*j+1] * f1 +
                          covariance_matrix[3*i,   3*j+2] * f2)
                delta1 = (covariance_matrix[3*i+1, 3*j]   * f0 +
                          covariance_matrix[3*i+1, 3*j+1] * f1 +
                          covariance_matrix[3*i+1, 3*j+2] * f2)
                delta2 = (covariance_matrix[3*i+2, 3*j]   * f0 +
                          covariance_matrix[3*i+2, 3*j+1] * f1 +
                          covariance_matrix[3*i+2, 3*j+2] * f2)
                s = sqrt(delta0*delta0 + delta1*delta1 + delta2*delta2)
                perturbation_matrix[i, j] += s

    # Normalize the perturbation matrix.
    sum_val = 0.0
    for i in range(m):
        for j in range(n):
            sum_val += perturbation_matrix[i, j]
    if sum_val != 0.0:
        for i in range(m):
            for j in range(n):
                perturbation_matrix[i, j] /= sum_val

    return perturbation_matrix


@cython.boundscheck(False)
@cython.wraparound(False)
@timeit
@memprofit
def _td_perturbation_matrix(np.ndarray[double, ndim=2] ccf, bint normalize=True) -> np.ndarray:
    """
    Calculate the perturbation matrix from a covariance matrix (or Hessian) on the CPU.
    
    The input covariance matrix 'ccf' is expected to have shape (3*m, 3*n).
    The function computes the perturbation value for each block:
    
        perturbation_matrix[i,j] = sqrt( sum_{a=0}^{2} sum_{b=0}^{2} (ccf[3*i+a, 3*j+b])^2 )
    
    If normalize is True, the output matrix is normalized so that its total sum equals 1.
    
    Parameters
    ----------
    ccf : np.ndarray[double, ndim=2]
        The input covariance matrix with shape (3*m, 3*n).
    normalize : bool, optional
        Whether to normalize the output perturbation matrix (default True).
    
    Returns
    -------
    perturbation_matrix : np.ndarray
        An (m, n) matrix of perturbation values.
    """
    cdef int m = ccf.shape[0] // 3
    cdef int n = ccf.shape[1] // 3
    cdef int i, j, a, b
    cdef double temp, sum_val = 0.0
    cdef np.ndarray[double, ndim=2] perturbation_matrix = np.empty((m, n), dtype=np.float64)
    
    # Loop over each block (i,j)
    for i in range(m):
        for j in range(n):
            temp = 0.0
            for a in range(3):
                for b in range(3):
                    temp += ccf[3*i + a, 3*j + b] * ccf[3*i + a, 3*j + b]
            perturbation_matrix[i, j] = sqrt(temp)
            sum_val += perturbation_matrix[i, j]
    
    if normalize and sum_val != 0.0:
        for i in range(m):
            for j in range(n):
                perturbation_matrix[i, j] /= sum_val
    
    return perturbation_matrix


    