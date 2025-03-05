"""
===============================================================================
File: test_rcmath.py
Description:
    This file contains unit tests for the 'rcmath' module from the 
    reforge.rfgmath package. The tests compare the outputs of the new 
    implementations with their legacy counterparts for various mathematical 
    functions, including Hessian calculation and perturbation matrices.

Usage:
    Run the tests with pytest:
        pytest -v tests/test_rcmath.py

Requirements:
    - NumPy
    - pytest
    - CuPy (for GPU tests; tests will be skipped if not installed)

Author: DY
Date: 2025-02-27
===============================================================================
"""

import numpy as np
import pytest
from reforge.rfgmath import rcmath, legacy
from reforge.utils import logger

# Set a fixed random seed for reproducibility of the tests.
np.random.seed(42)


def test_hessian():
    """
    Test the calculation of the Hessian matrix.

    This test compares the output of the '_calculate_hessian' function between
    the legacy and the new implementation. It performs the following steps:
      - Generates random arrays for x, y, and z coordinates.
      - Sets test parameters including the number of residues (n), cutoff,
        spring constant, and a distance dependence exponent (dd).
      - Computes the Hessian using both the legacy and new implementations.
      - Asserts that the results are almost identical within a tight tolerance.

    Returns:
        None
    """
    n = 50
    x = np.random.rand(n)
    y = np.random.rand(n)
    z = np.random.rand(n)
    vec = np.array((x, y, z)).T
    cutoff = 12
    spring_constant = 1000
    dd = 0
    legacy_result = legacy.calculate_hessian(n, x, y, z, cutoff, spring_constant, dd)
    new_result = rcmath.calculate_hessian(n, x, y, z, cutoff, spring_constant, dd)
    vec_result = rcmath.hessian(vec, cutoff, spring_constant, dd)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(vec_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_perturbation_matrix_old():
    """
    Compare the legacy and new implementations of the old perturbation matrix function.

    This test:
      - Creates a symmetric covariance matrix from a random matrix.
      - Computes the perturbation matrix using the '_perturbation_matrix_old'
        function from both the legacy and new implementations.
      - Asserts that the outputs are nearly identical within the specified tolerance.

    Returns:
        None
    """
    m = 200  # number of residues
    # Create a symmetric covariance matrix of shape (3*m, 3*m)
    A = np.random.rand(3 * m, 3 * m)
    covmat = (A + A.T) / 2
    # legacy_result = legacy.perturbation_matrix_old(covmat, m)
    legacy_result = legacy.calcperturbMat(covmat, m)
    new_result = rcmath.perturbation_matrix_old(covmat, m)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-4, atol=1e-4)


def test_perturbation_matrix_old2():
    """
    Compare the CPU-based perturbation matrix outputs between legacy and new implementations.

    This test:
      - Generates a symmetric covariance matrix.
      - Computes the perturbation matrix using the legacy CPU function and the
        new perturbation matrix function.
      - Verifies that the results match within a tight numerical tolerance.

    Returns:
        None
    """
    m = 200
    A = np.random.rand(3 * m, 3 * m)
    covmat = (A + A.T) / 2
    legacy_result = legacy.perturbation_matrix_cpu(covmat)
    new_result = rcmath.perturbation_matrix_old2(covmat)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_td_perturbation_matrix():
    """
    Compare the time-dependent perturbation matrix outputs between legacy and new implementations.

    This test:
      - Constructs a symmetric covariance matrix.
      - Computes the time-dependent perturbation matrix with normalization
        using the legacy CPU function and the new implementation.
      - Asserts that both results are almost identical within the specified tolerances.

    Returns:
        None
    """
    m = 20
    nt = 100
    A = np.random.rand(3 * m, 3 * m, nt)
    covmat = (A + A.T) / 2
    legacy_result = legacy.td_perturbation_matrix_cpu(covmat, normalize=True)
    new_result = rcmath.td_perturbation_matrix(covmat, normalize=True)
    np.testing.assert_allclose(new_result, legacy_result, rtol=1e-6, atol=1e-6)


def test_perturbation_matrix_old_new():
    """
    Compare the two new implementations of the perturbation matrix.

    This test compares the output of the old perturbation matrix function
    (as implemented in the new module) with the new perturbation matrix function.
    The test verifies that the two approaches yield nearly identical results
    within a slightly relaxed tolerance.

    Returns:
        None
    """
    m = 200
    A = np.random.rand(3 * m, 3 * m)
    covmat = (A + A.T) / 2
    old_result = rcmath.perturbation_matrix_old(covmat, m)
    new_result = rcmath.perturbation_matrix(covmat)
    np.testing.assert_allclose(new_result, old_result, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__])

