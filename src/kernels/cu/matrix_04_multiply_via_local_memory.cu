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
    __shared__ float a_buf[GROUP_SIZE_X][GROUP_SIZE_X];
    __shared__ float b_buf[GROUP_SIZE_X][GROUP_SIZE_X];

    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        float acc = 0;

        for (unsigned int tile = 0; tile < (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X; tile++) {
            const unsigned int a_x = tile * GROUP_SIZE_X + threadIdx.x;
            const unsigned int b_y = tile * GROUP_SIZE_X + threadIdx.y;

            a_buf[threadIdx.y][threadIdx.x] = a_x < k ? a[y * k + a_x] : 0;
            b_buf[threadIdx.y][threadIdx.x] = b_y < k ? b[b_y * w + x] : 0;

            __syncthreads();

            for (unsigned int i = 0; i < GROUP_SIZE_X; i++) {
                acc += a_buf[threadIdx.y][i] * b_buf[i][threadIdx.x];
            }

            __syncthreads();
        }

        c[y * w + x] = acc;
    }
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
