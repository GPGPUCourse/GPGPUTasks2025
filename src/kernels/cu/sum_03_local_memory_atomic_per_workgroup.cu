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
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;    
    __shared__ unsigned int local_data[GROUP_SIZE];


    if (index >= n / LOAD_K_VALUES_PER_ITEM) {
        local_data[threadIdx.x] = 0; // паддим память и не тратим время
        return;
    }

    unsigned int my_sum = 0;
    for (unsigned int i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        my_sum += a[i * (n/LOAD_K_VALUES_PER_ITEM) + index];
    }
    local_data[threadIdx.x] = my_sum;

    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int block_sum = 0;
        for (unsigned int k = 0; k < blockDim.x; ++k) {
            block_sum += local_data[k];
        }
        atomicAdd(sum, block_sum);
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
