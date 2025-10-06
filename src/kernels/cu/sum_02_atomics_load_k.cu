#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void sum_02_atomics_load_k(
    const unsigned int* a,
          unsigned int* sum,
          unsigned int  n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n / LOAD_K_VALUES_PER_ITEM) {
        return;
    }

    unsigned int my_sum = 0;
    for (unsigned int i = 0; i < LOAD_K_VALUES_PER_ITEM; ++i) {
        my_sum += a[i * (n/LOAD_K_VALUES_PER_ITEM) + index];
    }

    atomicAdd(sum, my_sum);
}

namespace cuda {
void sum_02_atomics_load_k(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_02_atomics_load_k<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
