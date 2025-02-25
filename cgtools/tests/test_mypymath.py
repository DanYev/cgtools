import os
import numpy as np
import pytest
from cgtools._actual_math import mypymath


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


def test_sfft_corr():
    """
    Test that _sfft_corr returns a correlation function that
    matches the manually computed sliding average correlation.
    """
    n_coords = 2
    n_samples = 128
    ntmax = 64
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)  
    x_centered = x - np.mean(x, axis=-1, keepdims=True)
    y_centered = y - np.mean(y, axis=-1, keepdims=True)   
    corr_fft = mypymath._sfft_corr(x, y, ntmax=ntmax, center=True, loop=True, dtype=np.float64)  
    ref_corr = np.empty((n_coords, n_coords, n_samples), dtype=np.float64) # Manually compute the sliding average correlation.
    for i in range(n_coords):
        for j in range(n_coords):
            for t in range(n_samples):
                ref_corr[i, j, t] = np.average(x_centered[i, t:] * y_centered[j, :n_samples-t])
    ref_corr = ref_corr[:,:,:ntmax]
    np.testing.assert_allclose(corr_fft, ref_corr, rtol=1e-10, atol=1e-10)    


def test_pfft_corr():
    """
    Test that _pfft_corr returns same result as _sfft_corr
    """
    n_coords = 2
    n_samples = 128
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    ntmax = 64
    corr_par = mypymath._pfft_corr(x, y, ntmax=ntmax, center=True, dtype=np.float64)
    corr_ser = mypymath._sfft_corr(x, y, ntmax=ntmax, center=True, loop=True, dtype=np.float64)
    np.testing.assert_allclose(corr_par, corr_ser, rtol=1e-10, atol=1e-10) 


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


@pytest.mark.skipif(cp is None, reason="CuPy is not installed")
def test_gfft_corr():
    """
    Test that _gfft_corr returns same result as _sfft_corr
    """
    n_coords = 2
    n_samples = 128
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    ntmax = 64
    corr_gpu = mypymath._gfft_corr(x, y, ntmax=ntmax, center=True, dtype=cp.float64)
    corr = corr_gpu.get()
    corr_ser = mypymath._sfft_corr(x, y, ntmax=ntmax, center=True, loop=True, dtype=np.float64)
    np.testing.assert_allclose(corr, corr_ser, rtol=1e-10, atol=1e-10) 


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

