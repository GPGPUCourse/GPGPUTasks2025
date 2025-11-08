#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ float shared_memory_a[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    __shared__ float shared_memory_b[GROUP_SIZE_Y + 1][GROUP_SIZE_X];

    float sum = 0;

    for (unsigned int i = 0; i < k; i += GROUP_SIZE_X) {
        if (row < h && col < w && i + threadIdx.x < k) {
            shared_memory_a[threadIdx.y][threadIdx.x] = a[row * k + i + threadIdx.x];
        }
        if (row < h && col < w && i + threadIdx.y < k) {
            shared_memory_b[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * w + col];
        }

        __syncthreads();

        for (unsigned int j = 0; j < GROUP_SIZE_Y; j++) {
            sum += shared_memory_a[threadIdx.y][j] * shared_memory_b[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < h && col < w) {
        c[row * w + col] = sum;
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
