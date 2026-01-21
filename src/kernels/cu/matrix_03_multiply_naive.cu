#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_naive(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int kk)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..w
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; // 0..h

    if (i > w) return;
    if (j > h) return;

    const uint c_ix = j*w + i;

    double acc = 0;
    for (uint k = 0; k < kk; ++k) {
        acc += a[j*kk + k] * b[k*w + i];
    }

    c[c_ix] = acc;
    // if (i == 0) printf("i=%d j=%d acc=%f out=%d\n", i, j, acc, c_ix);
}

namespace cuda {
void matrix_multiply_naive(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_naive<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
