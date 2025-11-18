#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_accumulate(
    unsigned int* out,
    const unsigned int* reduction_buffer,
    unsigned int n)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index > n)
        return;

    uint result = 0;
    uint offset = 0;
    uint size = n;
    while (true) {
        if (index % 2 == 0) {
            result += reduction_buffer[offset+index];
        }
        if (index == 0)
            break;
        offset += size;
        size /= 2;
        index = (index-1) / 2;
    }

    const uint index1 = blockIdx.x * blockDim.x + threadIdx.x;
    // reduction_buffer[index1] = result;
    out[index1] = result;
}

namespace cuda {
void prefix_sum_03_accumulate(const gpu::WorkSize &workSize,
                              gpu::gpu_mem_32u &result, const gpu::gpu_mem_32u &reduction_buffer, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_accumulate<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(result.cuptr(), reduction_buffer.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
