#include <device_launch_parameters.h>
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
    __shared__ float local_data_A[GROUP_SIZE];
    __shared__ float local_data_B[GROUP_SIZE];

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int index = y * w + x;
    const unsigned int local_index = threadIdx.x + threadIdx.y * blockDim.x;

    c[index] = 0;

    for (unsigned int i = 0; i < k / blockDim.x; ++i) {
        local_data_A[local_index] = a[y * k + threadIdx.x + i * blockDim.x];
        local_data_B[local_index] = b[(threadIdx.y + i * blockDim.y) * w + x];

        __syncthreads();

        for (unsigned int j = 0; j < blockDim.x; ++j) {
            c[index] += local_data_A[threadIdx.y * blockDim.x + j] * local_data_B[threadIdx.x + j * blockDim.y];
        }

        __syncthreads();
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
