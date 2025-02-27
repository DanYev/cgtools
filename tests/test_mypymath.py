import os
import numpy as np
import pytest
from reforge.actual_math import mypymath


def test_covariance_matrix():
    """
    Test that covariance_matrix returns an array with the correct shape
    and expected properties.
    """
    # Create a simple positions array: shape (n_coords, n_frames)
    positions = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]], dtype=np.float64)
    nt = 10
    positions = np.tile(positions, (nt, nt))
    covmat = mypymath._covariance_matrix(positions, dtype=np.float64)
    # Expect a (3, 3) covariance matrix when rowvar=True.
    assert covmat.shape == (3*nt, 3*nt)
    # For this degenerate data (linearly dependent), the determinant should be ~0.
    np.testing.assert_almost_equal(np.linalg.det(covmat), 0, decimal=5)


def test_sfft_corr():
    """
    Test that _sfft_corr returns a correlation function that
    matches the manually computed sliding average correlation.
    """
    n_coords = 10
    n_samples = 256
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
    n_coords = 10
    n_samples = 256
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    ntmax = 64
    corr_par = mypymath._pfft_corr(x, y, ntmax=ntmax, center=True, dtype=np.float64)
    corr_ser = mypymath._sfft_corr(x, y, ntmax=ntmax, center=True, loop=True, dtype=np.float64)
    np.testing.assert_allclose(corr_par, corr_ser, rtol=1e-10, atol=1e-10) 


def test_ccf():
    """
    Test that ccf returns a cross-correlation function that matches the 
    manually computed sliding average correlation.
    """
    n_coords = 10
    n_samples = 256
    n_seg = 4  # number of segments
    x = np.random.rand(n_coords, n_samples).astype(np.float64)
    y = np.random.rand(n_coords, n_samples).astype(np.float64)
    segments_x = np.array_split(x, n_seg, axis=-1)
    segments_y = np.array_split(y, n_seg, axis=-1)
    manual_corr_sum = None
    for seg_x, seg_y in zip(segments_x, segments_y):
        x_centered = seg_x - np.mean(seg_x, axis=-1, keepdims=True)
        y_centered = seg_y - np.mean(seg_y, axis=-1, keepdims=True)
        nt_seg = seg_x.shape[-1]
        ntmax_seg = (nt_seg + 1) // 2
        manual_corr_seg = np.empty((n_coords, n_coords, ntmax_seg), dtype=np.float64)
        for i in range(n_coords):
            for j in range(n_coords):
                for tau in range(ntmax_seg):
                    window = x_centered[i, tau:nt_seg] * y_centered[j, :nt_seg-tau ]
                    manual_corr_seg[i, j, tau] = np.average(window)
        if manual_corr_sum is None:
            manual_corr_sum = manual_corr_seg
        else:
            manual_corr_sum += manual_corr_seg
    manual_ccf = manual_corr_sum / n_seg
    par_ccf = mypymath.ccf(x, y, ntmax=None, n=n_seg, mode='parallel', center=True, dtype=np.float64)
    ser_ccf = mypymath.ccf(x, y, ntmax=None, n=n_seg, mode='serial', center=True, dtype=np.float64)
    np.testing.assert_allclose(manual_ccf, par_ccf, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(manual_ccf, ser_ccf, rtol=1e-6, atol=1e-6)


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
    n_coords = 10
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
    N = 200
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

