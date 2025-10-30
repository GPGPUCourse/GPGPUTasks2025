#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../wrappers.h"
#include "helpers/rassert.cu"

__global__ void radix_post(const unsigned int* a, const unsigned int* b, unsigned int* c, unsigned int offset, unsigned int n) {
    int x = threadIdx.x;
    int i = blockIdx.x * blockDim.x + x;
    unsigned int zero_count = b[n - 1];
    if (i < n) {
        int t = ((a[i] >> offset) & 1);
        int cind = (1 - t) * (b[i] - 1) + t * (zero_count + i - b[i]);
        c[cind] = a[i];
    }
}

namespace cuda {
void radix_post(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, gpuptr::u32 c, unsigned int offset, unsigned int n) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_post<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), offset, n);
    CUDA_CHECK_KERNEL(stream);
}
}  // namespace cuda
