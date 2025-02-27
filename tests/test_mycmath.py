import numpy as np
import pytest
from reforge.actual_math import mycmath, legacy
# Set a seed for reproducibility
np.random.seed(42)


def test_calculate_hessian():
    """
    Compare the output of calculate_hessian between legacy and new implementation.
    """
    resnum = 50
    x = np.random.rand(resnum)
    y = np.random.rand(resnum)
    z = np.random.rand(resnum)
    cutoff = 12
    spring_constant = 1000
    dd = 0
    legacy_result = legacy._calculate_hessian(resnum, x, y, z, cutoff, spring_constant, dd)
    new_result = mycmath._calculate_hessian(resnum, x, y, z, cutoff, spring_constant, dd)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_perturbation_matrix_old():
    """
    Compare the _perturbation_matrix_old function outputs between legacy and new implementations.
    """
    m = 50  # number of residues
    # Create a symmetric covariance matrix of shape (3*m, 3*m)
    A = np.random.rand(3 * m, 3 * m)
    covmat = (A + A.T) / 2
    legacy_result = legacy._perturbation_matrix_old(covmat, m)
    new_result = mycmath._perturbation_matrix_old(covmat, m)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_perturbation_matrix():
    """
    Compare the _perturbation_matrix_cpu function outputs between legacy and new implementations.
    """
    m = 50
    A = np.random.rand(3 * m, 3 * m)
    covmat = (A + A.T) / 2
    legacy_result = legacy._perturbation_matrix_cpu(covmat)
    new_result = mycmath._perturbation_matrix(covmat)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_td_perturbation_matrix():
    """
    Compare the _td_perturbation_matrix_cpu outputs between legacy and new implementations.
    """
    m = 50
    A = np.random.rand(3 * m, 3 * m)
    covmat = (A + A.T) / 2
    legacy_result = legacy._td_perturbation_matrix_cpu(covmat, normalize=True)
    new_result = mycmath._td_perturbation_matrix(covmat, normalize=True)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_perturbation_matrix_old_new():
    """
    Compare the _perturbation_matrix_cpu function outputs between legacy and new implementations.
    """
    m = 50
    A = np.random.rand(3 * m, 3 * m)
    covmat = (A + A.T) / 2
    old_result = mycmath._perturbation_matrix_old(covmat, m)
    new_result = mycmath._perturbation_matrix(covmat)
    np.testing.assert_allclose(new_result, old_result, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
