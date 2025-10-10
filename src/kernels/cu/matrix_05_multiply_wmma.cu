#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

// Include WMMA header with nvcuda::wmma namespace
#include <mma.h>
using namespace nvcuda;

#define TILE 16
__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    curassert(blockDim.x == 16, 435234124321);
    curassert(blockDim.y == 2, 5432343123241);

    // Each block computes one TILE x TILE tile of C
    // Starting from row0 x col0 element
    const unsigned int col0 = blockIdx.x*TILE;
    const unsigned int row0 = blockIdx.y*TILE/2; // bug: *TILE

    const unsigned int inWarpIndex = 16 * threadIdx.y + threadIdx.x;

    // Shared tiles (aligned for wmma::load)
    __shared__ __align__(16) half As[TILE * TILE];
    __shared__ __align__(16) half Bs[TILE * TILE];

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> tileA;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::row_major> tileB;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> accumulator;

    wmma::fill_fragment(accumulator, 0.0f);

    for (unsigned int k0 = 0; k0 < k; k0 += TILE) {
        for (int inTileIndex = inWarpIndex; inTileIndex < TILE * TILE; inTileIndex += 32) {
            int tileRow = inTileIndex / TILE;
            int tileCol = inTileIndex % TILE;

            unsigned int aRow = row0 + tileRow;
            unsigned int aCol = k0 + tileCol;
            unsigned int bRow = k0 + tileRow;
            unsigned int bCol = col0 + tileCol;

            As[inTileIndex] = __float2half_rn(a[aRow * k + aCol]);
            Bs[inTileIndex] = __float2half_rn(b[bRow * w + bCol]);
        }

        __syncwarp();
        wmma::load_matrix_sync(tileA, As, TILE);
        wmma::load_matrix_sync(tileB, Bs, TILE);
        wmma::mma_sync(accumulator, tileA, tileB, accumulator);
        __syncwarp();
    }

    wmma::store_matrix_sync(c + row0 * w + col0, accumulator, w, wmma::mem_row_major);
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

