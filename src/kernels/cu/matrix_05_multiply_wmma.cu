#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
#include <mma.h>
using namespace nvcuda;

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    constexpr unsigned int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr unsigned int BLOCK_M = 32, BLOCK_N = 32;
    constexpr unsigned int as_size = BLOCK_M * WMMA_K, bs_size = WMMA_K * BLOCK_N;
    const unsigned int threads_per_block = blockDim.x * blockDim.y;
    const unsigned int thrd = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int warp = thrd / warpSize;
    const unsigned int warp_row = warp / 2;
    const unsigned int warp_col = warp % 2;
    const unsigned int row_start = blockIdx.y * BLOCK_M;
    const unsigned int col_start = blockIdx.x * BLOCK_N;
    const unsigned int res_row_start = row_start + warp_row * WMMA_M;
    const unsigned int res_col_start = col_start + warp_col * WMMA_N;
    
    __shared__ half as[as_size];
    __shared__ half bs[bs_size];
    const half* as_ptr = &as[warp_row * WMMA_M * WMMA_K];
    const half* bs_ptr = &bs[warp_col * WMMA_N * WMMA_K];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    #pragma unroll
    for (unsigned int i = 0; i < k; i += WMMA_K) {
        #pragma unroll
        for (unsigned int t = thrd; t < as_size; t += threads_per_block) {
            unsigned int row = t / WMMA_K;
            unsigned int col = t % WMMA_K;
            unsigned int a_row = row_start + row;
            unsigned int a_col = i + col;
            float val = 0;
            if (a_row < h && a_col < k)
                val = __ldg(&a[a_row * k + a_col]);
            as[row * WMMA_K + col] = __float2half_rn(val);
        }

        #pragma unroll
        for (unsigned int t = thrd; t < bs_size; t += threads_per_block) {
            unsigned int row = t / BLOCK_N;
            unsigned int col = t % BLOCK_N;
            unsigned int b_row = i + row;
            unsigned int b_col = col_start + col;
            float val = 0;
            if (b_row < k && b_col < w)
                val = __ldg(&b[b_row * w + b_col]);
            bs[row + col * WMMA_K] = __float2half_rn(val);
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, as_ptr, WMMA_K);
        wmma::load_matrix_sync(b_frag, bs_ptr, WMMA_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync(c + res_row_start * w + res_col_start, c_frag, w, wmma::mem_row_major);
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda

