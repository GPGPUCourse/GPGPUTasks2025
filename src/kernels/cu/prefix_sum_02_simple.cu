#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void prefix_sum_simple(
    const unsigned int* a, // input n
          unsigned int* c, // output; n
    unsigned int n)
{
    int i = threadIdx.x;
    int glob_i = blockIdx.x * blockDim.x + i;
    if (glob_i != 0) {
        return;
    }
    
    unsigned int acc = 0;
    for (int j = 0; j < n; ++j) {
        acc += a[j];
        c[j] = acc;
    }
}

namespace cuda {
void prefix_sum_simple(const gpu::WorkSize &workSize, 
    const gpu::gpu_mem_32u &a, unsigned int abase,
    gpu::gpu_mem_32u &c, unsigned int cbase,
    unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefix_sum_simple<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr() + abase, c.cuptr() + cbase, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
