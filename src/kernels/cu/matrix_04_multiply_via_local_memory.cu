#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_multiply_via_local_memory(
    const float* a, // rows=h x cols=k
    const float* b, // rows=k x cols=w
    float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int row = blockIdx.y * GROUP_SIZE_Y + ty;
    const unsigned int col = blockIdx.x * GROUP_SIZE_X + tx;

    __shared__ float Asub[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float Bsub[GROUP_SIZE_X][GROUP_SIZE_X];

    float acc = 0;

    const unsigned int numTiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (unsigned int t = 0; t < numTiles; ++t) {
        unsigned int a_col_base = t * GROUP_SIZE_X;
        unsigned int b_row_base = t * GROUP_SIZE_X;

        for (unsigned int i = tx; i < GROUP_SIZE_X; i += blockDim.x) {
            unsigned int a_col = a_col_base + i;
            float aval = (row < h && a_col < k) ? a[row * k + a_col] : 0.0f;
            Asub[ty][i] = aval;
        }

        for (unsigned int i = ty; i < GROUP_SIZE_X; i += blockDim.y) {
            unsigned int b_row = b_row_base + i;
            float bval = (b_row < k && col < w) ? b[b_row * w + col] : 0.0f;
            Bsub[i][tx] = bval;
        }

        __syncthreads();

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i) {
            acc += Asub[ty][i] * Bsub[i][tx];
        }

        __syncthreads();
    }

    if (row < h && col < w) {
        c[row * w + col] = acc;
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
