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
    wmma::fragment<wmma::matrix_a, GROUP_SIZE_Y, GROUP_SIZE_X, GROUP_SIZE_Y, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, GROUP_SIZE_Y, GROUP_SIZE_X, GROUP_SIZE_Y, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, GROUP_SIZE_Y, GROUP_SIZE_X, GROUP_SIZE_Y, float> c_frag;

    __shared__ __align__(16) half As[GROUP_SIZE_Y * GROUP_SIZE_X];
    __shared__ __align__(16) half Bs[GROUP_SIZE_Y * GROUP_SIZE_X];

    wmma::fill_fragment(c_frag, 0.0f);

    for (unsigned int i = 0; i < k; i += GROUP_SIZE_X) {
        for (unsigned int j = blockDim.x * threadIdx.y + threadIdx.x; j < GROUP_SIZE_X * GROUP_SIZE_Y; j += blockDim.x * blockDim.y) {
            unsigned int tile_row = j / GROUP_SIZE_X;
            unsigned int tile_col = j % GROUP_SIZE_X;
            
            unsigned int a_row = blockIdx.y * GROUP_SIZE_Y + tile_row;
            unsigned int a_col = i + tile_col;
            unsigned int b_row = i + tile_row;
            unsigned int b_col = blockIdx.x * GROUP_SIZE_X + tile_col;
            As[j] = __float2half_rn(a[a_row * k + a_col]);
            Bs[j] = __float2half_rn(b[b_row * w + b_col]);
        }

        wmma::load_matrix_sync(a_frag, As, GROUP_SIZE_X);
        wmma::load_matrix_sync(b_frag, Bs, GROUP_SIZE_X);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    unsigned int c_row = blockIdx.y * GROUP_SIZE_Y;
    unsigned int c_col = blockIdx.x * GROUP_SIZE_X;

    if (c_row < h && c_col < w) {
        wmma::store_matrix_sync(c + c_row * w + c_col, c_frag, w, wmma::mem_row_major);
    }
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

