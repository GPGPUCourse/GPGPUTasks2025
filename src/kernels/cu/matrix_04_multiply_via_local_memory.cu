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
    unsigned global_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned global_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned local_x = threadIdx.x;
    unsigned local_y = threadIdx.y;


    __shared__ float A_buffer[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float B_buffer[GROUP_SIZE_X][GROUP_SIZE_X];

    float sum = 0.0f;

    for (unsigned block_k = 0; block_k < k; block_k += GROUP_SIZE_X) {
        if (global_y < h && (block_k + local_x) < k) {
            A_buffer[local_y][local_x] = a[global_y * k + (block_k + local_x)];
        } else {
            A_buffer[local_y][local_x] = 0.0f;
        }

        if ((block_k + local_y) < k && global_x < w) {
            B_buffer[local_y][local_x] = b[(block_k + local_y) * w + global_x];
        } else {
            B_buffer[local_y][local_x] = 0.0f;
        }

        __syncthreads();

        for (unsigned tile_k = 0; tile_k < GROUP_SIZE_X; ++tile_k) {
            sum += A_buffer[local_y][tile_k] * B_buffer[tile_k][local_x];
        }

        __syncthreads();
    }

    if (global_y < h && global_x < w) {
        c[global_y * w + global_x] = sum;
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
