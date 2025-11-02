#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../wrappers.h"
#include "helpers/rassert.cu"

__global__ void copy(
    const unsigned int* a,
    unsigned int* b,
    unsigned int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i < n) {
        b[i] = a[i];
    } 
}

namespace cuda {
void copy(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::copy<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
