
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "../wrappers.h"

__global__ void prefixsum_post(
    unsigned int* b, // reduced buffer;
    unsigned int* c, // output
    unsigned int n)
{
    int i = threadIdx.x;
    int glob_i = blockIdx.x * blockDim.x + i;
    
    unsigned int offset = 0;
    if (blockIdx.x > 0) {
        offset = b[blockIdx.x - 1];
    }
    
    if (glob_i < n) {
        c[glob_i] = c[glob_i] + offset;
    }
}

namespace cuda {
void prefixsum_post(const gpu::WorkSize& workSize, gpuptr::u32 b, gpuptr::u32 c, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefixsum_post<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(b.cuptr(), c.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
