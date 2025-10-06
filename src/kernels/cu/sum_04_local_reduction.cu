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
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_index = threadIdx.x;
    __shared__ unsigned int local_data[GROUP_SIZE];
    if (index < n) {
        local_data[local_index] = 0;
        local_data[local_index] += a[index];
    } else {
        local_data[local_index] = 0;
    }
    __syncthreads();
    if (local_index == 0) {
        unsigned int local_sum = 0;
        for (int i = 0; i < GROUP_SIZE; ++i) {
            local_sum += local_data[i];
        }
        b[index / GROUP_SIZE] = local_sum;
    }
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
