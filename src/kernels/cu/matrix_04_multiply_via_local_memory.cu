#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float as[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float bs[GROUP_SIZE_X][GROUP_SIZE_X];
    const unsigned int i = blockIdx.x * GROUP_SIZE_X + threadIdx.x;
    const unsigned int j = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;
    
    float acc = 0;
    #pragma unroll
    for (unsigned int kk = 0; kk < k; kk += GROUP_SIZE_X) {
        if (j < h && (kk + threadIdx.x) < k)
            as[threadIdx.y][threadIdx.x] = a[j * k + (kk + threadIdx.x)];
        else
            as[threadIdx.y][threadIdx.x] = 0;

        if ((kk + threadIdx.y) < k && i < w)
            bs[threadIdx.y][threadIdx.x] = b[(kk + threadIdx.y) * w + i];
        else
            bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (unsigned int kkk = 0; kkk < GROUP_SIZE_X; ++kkk)
            acc += as[threadIdx.y][kkk] * bs[kkk][threadIdx.x];

        __syncthreads();
    }

    if (i < w && j < h)
        c[j * w + i] = acc;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
