#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned int* arr_in,
    unsigned int* buffer,
    unsigned int n,
    unsigned int offset)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        buffer[idx] = ~(arr_in[idx] >> offset) & 0x1;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& arr_in, gpu::gpu_mem_32u& buffer, unsigned int n, unsigned int offset)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        arr_in.cuptr(), buffer.cuptr(), n, offset);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda