#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void sum_03_local_memory_atomic_per_workgroup(
    const unsigned int* a,
    unsigned int* sum,
    unsigned int n)
{
    // Подсказки:
    // const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint local_index = threadIdx.x;
    // __shared__ unsigned int local_data[GROUP_SIZE];
    // __syncthreads();

    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];

    unsigned int partial_sum = 0;

    for (uint i = index; i < n; i += blockDim.x * gridDim.x) {
        partial_sum += a[i];
    }

    local_data[local_index] = partial_sum;
    __syncthreads();

    for (uint s = blockDim.x / 2; s > 0; s /= 2) {
        if (local_index < s) {
            local_data[local_index] += local_data[local_index + s];
        }
        __syncthreads();
    }

    if (local_index == 0) {
        atomicAdd(sum, local_data[0]);
    }
}

namespace cuda {
void sum_03_local_memory_atomic_per_workgroup(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_03_local_memory_atomic_per_workgroup<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
