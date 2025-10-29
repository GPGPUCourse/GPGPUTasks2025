#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
    const float *a, // rows=h x cols=k
    const float *b, // rows=k x cols=w
    float *c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float Asub[GROUP_SIZE_X][GROUP_SIZE_Y];
    __shared__ float Bsub[GROUP_SIZE_X][GROUP_SIZE_Y];

    float sum = 0.0f;

    unsigned int numTiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (unsigned int t = 0; t < numTiles; ++t) {
        unsigned int a_col = t * GROUP_SIZE_X + threadIdx.x;
        unsigned int b_row = t * GROUP_SIZE_Y + threadIdx.y;

        if (row < h && a_col < k) {
            Asub[threadIdx.y][threadIdx.x] = a[row * k + a_col];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < k && col < w) {
            Bsub[threadIdx.y][threadIdx.x] = b[b_row * w + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();


        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i)
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

        __syncthreads();
    }

    if (row < h && col < w)
        c[row * w + col] = sum;
}

namespace cuda {
    void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
                                          const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c,
                                          unsigned int w, unsigned int h, unsigned int k) {
        gpu::Context context;
        rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
        cudaStream_t stream = context.cudaStream();
        ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
            a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
        CUDA_CHECK_KERNEL(stream);
    }
} // namespace cuda
