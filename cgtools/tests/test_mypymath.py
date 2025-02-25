import os
import numpy as np
import pytest
from cgtools._actual_math import mypymath

# Test covariance_matrix function.
def test_covariance_matrix():
    """
    Test that covariance_matrix returns an array with the correct shape
    and expected properties.
    """
    # Create a simple positions array: shape (n_coords, n_frames)
    positions = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=np.float64)
    covmat = mypymath.covariance_matrix(positions, dtype=np.float64)
    # Expect a (3, 3) covariance matrix when rowvar=True.
    assert covmat.shape == (3, 3)
    # For this degenerate data (linearly dependent), the determinant should be ~0.
    np.testing.assert_almost_equal(np.linalg.det(covmat), 0, decimal=5)

# Test sfft_cpsd function.
def test_sfft_corr():
    """
    Test that _sfft_corr returns a correlation function with expected shape.
    """
    n_coords = 2
    n_samples = 128
    # Generate random signals.
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    # Specify ntmax.
    ntmax = 64
    # Call the function with loop=True.
    corr = mypymath._sfft_corr(x, y, ntmax=ntmax, center=True, loop=True, dtype=np.float64)
    # The expected shape is (n_coords, n_coords, ntmax).
    assert corr.shape == (n_coords, n_coords, ntmax)


# Test for the CPU version using sparse eigensolver.
def test_inverse_sparse_matrix_cpu():
    # Create a 10x10 diagonal matrix.
    N = 100
    diag_vals = np.linspace(1, 1e7, N)
    matrix = np.diag(diag_vals)
    # Call the function with k_singular=0 and n_modes=N so that all eigenvalues are inverted.
    inv_matrix = mypymath._inverse_sparse_matrix_cpu(matrix, k_singular=0, n_modes=N-1)
    # Expected inverse is the diagonal with reciprocals.
    expected_inv = np.diag(1.0 / diag_vals)
    np.testing.assert_allclose(inv_matrix, expected_inv, rtol=0, atol=1e-6)


# Skip GPU tests if CuPy is not installed.
try:
    import cupy as cp
    # import cupyx.scipy.sparse.linalg  # noqa: F401
except ImportError:
    cp = None

# Test _gfft_corr function only if CUDA is available.
@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
def test_gfft_corr():
    """
    Test that _gfft_corr returns a correlation function with expected shape.
    This test uses the GPU version and will be skipped if CUDA is not available.
    """
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("Cupy not installed")
    n_coords = 2
    n_samples = 128
    x = np.random.rand(n_coords, n_samples).astype(np.float32)
    y = np.random.rand(n_coords, n_samples).astype(np.float32)
    ntmax = 64
    corr = mypymath._gfft_corr(x, y, ntmax=ntmax, center=True, dtype=cp.float32)
    assert corr.shape == (n_coords, n_coords, ntmax)


@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
def test_inverse_sparse_matrix_gpu():
    # Create a 10x10 diagonal matrix.
    N = 200
    diag_vals = np.linspace(1, 10, N)
    matrix = np.diag(diag_vals)
    # Call the GPU version with k_singular=0 and n_modes=N.
    inv_matrix_gpu = mypymath._inverse_sparse_matrix_gpu(matrix, k_singular=0, n_modes=N//10, gpu_dtype=cp.float64)
    # Convert the result from CuPy array to a NumPy array.
    inv_matrix = cp.asnumpy(inv_matrix_gpu)
    expected_inv = mypymath._inverse_sparse_matrix_cpu(matrix, k_singular=0, n_modes=N//10)
    np.testing.assert_allclose(inv_matrix, expected_inv, rtol=0, atol=1e-6)


@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
def test_inverse_matrix_gpu():
    # Create a 10x10 diagonal matrix.
    N = 20
    diag_vals = np.linspace(1, 10, N)
    matrix = np.diag(diag_vals)
    # Call the GPU inverse using cupy.linalg.eigh with k_singular=0 and n_modes=N.
    inv_matrix_gpu = mypymath._inverse_matrix_gpu(matrix, k_singular=0, n_modes=N)
    # Convert to NumPy array.
    inv_matrix = cp.asnumpy(inv_matrix_gpu)
    expected_inv = np.diag(1.0 / diag_vals)
    np.testing.assert_allclose(inv_matrix, expected_inv, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
