###########################################################################################
# GPU kernels
###########################################################################################

# DFI KERNEL
dfi_kernel_code = """
extern "C" __global__ void dfi_kernel(const float* cov, const float* forces, const int resnum, float *result) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int twx = blockDim.x, twy = blockDim.y;
    
    __shared__ float f[3];
    
    // Load forces array into shared memory
    if (tx < 3) {
        f[tx] = forces[tx];
    }
    __syncthreads();
    

    if (bx < resnum && by < resnum){
        float sum_ij = 0;
        // Compute partial sum of this tile
        for (int i = 0; i < twy; i++){
            float partial_sum = 0;
            for (int j = 0; j < twx; j++){
                int row = by * twy + i;
                int col = bx * twx + j;
                int index = row * 3 * resnum + col;
                partial_sum += cov[index] * forces[j] * cov[index] * forces[j];
            }
            sum_ij += partial_sum;
        }
        sum_ij = sqrtf(sum_ij);
        __syncthreads();
        result[by*resnum + bx] = sum_ij;
    }
    
};
"""
dfi_kernel = cp.RawKernel(dfi_kernel_code, "dfi_kernel")


# INVERSE KERNEL
inverse_kernel_code = """
extern "C" __global__ void inverse_kernel(const float* cov) {

    
};
"""
inverse_kernel = cp.RawKernel(dfi_kernel_code, "inverse_kernel")