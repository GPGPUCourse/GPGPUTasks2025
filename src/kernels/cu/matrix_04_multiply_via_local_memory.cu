#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#define LOC_SIZE GROUP_SIZE_X

// precondition: blockDim.x == blockDim.y
__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    unsigned int sz = GROUP_SIZE_X; // = GROUP_SIZE_Y

    unsigned int col = blockIdx.x * sz + threadIdx.x;
    unsigned int row = blockIdx.y * sz + threadIdx.y;
    float acc = 0.0;

    __shared__ float a_loc[LOC_SIZE * LOC_SIZE];
    __shared__ float b_loc[LOC_SIZE * LOC_SIZE];

    for (unsigned int i = 0; i < k; i += sz) {
        // stage 01: copy
        if (col < w && row < h) {
            unsigned int ai = i + threadIdx.x;
            unsigned int aj = row;
            a_loc[threadIdx.y * sz + threadIdx.x] = a[aj * k + ai]; // copying as is

            unsigned int bi = col;
            unsigned int bj = i + threadIdx.y;
            b_loc[threadIdx.y * sz + threadIdx.x] = b[bj * w + bi]; // copying as is
        }

        __syncthreads();

        // stage 02: multiplying and accumulating
        if (col < w && row < h) {
            #pragma unroll
            for (unsigned int i_loc = 0; i_loc < sz; ++i_loc) {
                acc += a_loc[threadIdx.y * sz + i_loc] * b_loc[i_loc * sz + threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (col < w && row < h) {
        c[row * w + col] = acc;
    }
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
