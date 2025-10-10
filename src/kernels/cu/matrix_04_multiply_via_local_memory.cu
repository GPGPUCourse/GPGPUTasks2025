#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define TILE 16

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    // Each block computes one TILE×TILE tile of C
    const unsigned int row = blockIdx.y * TILE + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE + threadIdx.x;

    // Shared tiles
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;

    // Number of tiles along K dimension
    const unsigned int numTiles = k / TILE;

    for (unsigned int t = 0; t < numTiles; ++t) {
        // Coalesced loads into shared memory
        // A tile: row fixed, advancing along K
        As[threadIdx.y][threadIdx.x] = a[row * k + (t * TILE + threadIdx.x)];
        // B tile: col fixed, advancing along K
        Bs[threadIdx.y][threadIdx.x] = b[(t * TILE + threadIdx.y) * w + col];

        __syncthreads(); // Ensure tile is fully loaded

        // Compute partial dot product for this tile
        #pragma unroll
        for (int n = 0; n < TILE; ++n) {
            acc += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }

        __syncthreads(); // Avoid racing next tile loads
    }

    // Write the result
    c[row * w + col] = acc;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
