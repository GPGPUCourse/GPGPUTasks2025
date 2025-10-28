#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"
#include "../wrappers.h"

__global__ void prefixsum_main(
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
void prefixsum_main(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 c, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::prefixsum_main<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), c.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
