#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../wrappers.h"
#include "helpers/rassert.cu"

__global__ void fill(
    unsigned int* b,
    unsigned int x,
    unsigned int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // printf("b[%d]=%d(n=%d)\n", i, x, n);
        b[i] = x;
    } 
}

namespace cuda {
void fill(const gpu::WorkSize& workSize, const gpuptr::u32 b, unsigned int x, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::fill<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(b.cuptr(), x, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
