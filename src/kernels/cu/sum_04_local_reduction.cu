#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int n)
{
    __shared__ unsigned int local_data[GROUP_SIZE];

    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_index = threadIdx.x;
    const unsigned int block_index = blockIdx.x;
    const unsigned int BLOCK_SIZE = (n + LOAD_K_VALUES_PER_ITEM - 1) / LOAD_K_VALUES_PER_ITEM;

    unsigned int thread_sum = 0;
    if (index < BLOCK_SIZE) {
        for (unsigned int i = 0; i < LOAD_K_VALUES_PER_ITEM; i++) {
            thread_sum += a[i * BLOCK_SIZE + index];
        }
    }

    local_data[local_index] = thread_sum;
    __syncthreads();

    if (local_index != 0) {
        return;
    }
    unsigned int block_sum = 0;
    for (unsigned int i = 0; i < GROUP_SIZE; i++) {
        block_sum += local_data[i];
    }

    b[block_index] = block_sum;
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
