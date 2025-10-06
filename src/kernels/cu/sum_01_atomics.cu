#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void sum_01_atomics(
    const unsigned int* a,
          unsigned int* sum,
          unsigned int  n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n)
        return;

    atomicAdd(sum, a[index]);
}

namespace cuda {
void sum_01_atomics(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 2435231412, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_01_atomics<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
