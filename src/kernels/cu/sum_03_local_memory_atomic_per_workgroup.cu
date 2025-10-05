#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void sum_03_local_memory_atomic_per_workgroup(
    const unsigned int* a,
    unsigned int* sum,
    unsigned int  n)
{
    // Подсказки:
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];
    // __syncthreads();

    // TODO
    if (index < n)
        local_data[local_index] = a[index];
    else 
        local_data[local_index] = 0;
    __syncthreads();

    if (local_index == 0) {
        unsigned int my_sum = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; ++i) {
            my_sum += local_data[i];
        }
        atomicAdd(sum, my_sum);
    }
}

namespace cuda {
void sum_03_local_memory_atomic_per_workgroup(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_03_local_memory_atomic_per_workgroup<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
