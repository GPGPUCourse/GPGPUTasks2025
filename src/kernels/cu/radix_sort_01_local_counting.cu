#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void radix_sort_01_local_counting(
    const unsigned int* arr,
    unsigned int* prefix,
    unsigned int pow2,
    unsigned int n)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        prefix[index] = (arr[index] & (1 << pow2)) == 0;
    }
}

namespace cuda {
void radix_sort_01_local_counting(
    const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& arr,
    gpu::gpu_mem_32u& prefix,
    unsigned int pow2,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        arr.cuptr(), prefix.cuptr(), pow2, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
