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
                       const __half* a, // rows=h x cols=k
                       const __half* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    const unsigned int warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const unsigned int warp_y = blockIdx.y * blockDim.y + threadIdx.y;

    const int wmma_tile = 16;

    const unsigned int row_base = warp_y * wmma_tile;
    const unsigned int col_base = warp_x * wmma_tile;

    if (row_base >= h || col_base >= w) return;

    wmma::fragment<wmma::matrix_a, wmma_tile, wmma_tile, wmma_tile, __half, wmma::row_major> a_tile;
    wmma::fragment<wmma::matrix_b, wmma_tile, wmma_tile, wmma_tile, __half, wmma::row_major> b_tile;
    wmma::fragment<wmma::accumulator, wmma_tile, wmma_tile, wmma_tile, float> acc;

    wmma::fill_fragment(acc, 0.);

    for (unsigned int tile_k = 0; tile_k < k; tile_k += wmma_tile)
    {
        wmma::load_matrix_sync(a_tile, a + row_base * k + tile_k, k);
        wmma::load_matrix_sync(b_tile, b + tile_k * w + col_base, w);

        wmma::mma_sync(acc, a_tile, b_tile, acc);
    }

    wmma::store_matrix_sync(c + row_base * w + col_base, acc, w, wmma::mem_row_major);
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_half &a, const gpu::gpu_mem_half &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda

