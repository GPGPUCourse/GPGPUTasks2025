#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{
    __shared__ unsigned int local_data[GROUP_SIZE];
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_index = threadIdx.x;

    unsigned int my_sum = 0;
    if (index < n / LOAD_K_VALUES_PER_ITEM) {
        for (unsigned int i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
            unsigned int id = i * (n/LOAD_K_VALUES_PER_ITEM) + index;
            if (id < n) {
                my_sum += a[id];
            }
        }
    }
    local_data[local_index] = my_sum;
    __syncthreads();

    if (local_index == 0) {
        for (unsigned int i = 1; i < GROUP_SIZE; ++i) {
            my_sum += local_data[i];
        }
        b[blockIdx.x] = my_sum;
    }
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
