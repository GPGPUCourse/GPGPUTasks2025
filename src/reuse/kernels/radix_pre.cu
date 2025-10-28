#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "../wrappers.h"
#include "helpers/rassert.cu"

__global__ void radix_pre(const unsigned int* a, unsigned int* b, unsigned int offset, unsigned int n) {
    int x = threadIdx.x;
    int i = blockIdx.x * blockDim.x + x;
    if (i < n) {
        int res = ((a[i] >> offset) & 1) ^ 1;
        // printf("%d %d %d\n", i, a[i], res);
        b[i] = res;
    }
}

namespace cuda {
void radix_pre(const gpu::WorkSize& workSize, const gpuptr::u32 a, gpuptr::u32 b, unsigned int offset, unsigned int n) {
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::radix_pre<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), offset, n);
    CUDA_CHECK_KERNEL(stream);
}
}  // namespace cuda
