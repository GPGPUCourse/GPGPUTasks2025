#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"
#include "helpers/rassert.cu"

__global__ void matrix_multiply_via_local_memory(
    const float* a, // rows=h x cols=k
    const float* b, // rows=k x cols=w
    float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    constexpr int SIZE = GROUP_SIZE_X;

    __shared__ float buffer_a[GROUP_SIZE];
    __shared__ float buffer_b[GROUP_SIZE];
    __shared__ float buffer_c[GROUP_SIZE];

    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;

    const int local_x = threadIdx.x;
    const int local_y = threadIdx.y;

    const int local_index = local_y * GROUP_SIZE_X + local_x;
    buffer_c[local_index] = 0;

    for (int ik = 0; ik < k / SIZE; ik++) {
        // initialize buffer arrays
        buffer_a[local_index] = 0;
        buffer_b[local_index] = 0;
        if (block_y + local_y < h && (ik * SIZE + local_x) < k) {
            buffer_a[local_index] = a[(block_y + local_y) * k + ik * SIZE + local_x];
        }
        if (ik * SIZE + local_y < k && block_x + local_x < w) {
            buffer_b[local_index] = b[(ik * SIZE + local_y) * w + block_x + local_x];
        }
        __syncthreads();

        // multiply
        for (int it = 0; it < SIZE; it++) {
            buffer_c[local_index] += buffer_a[local_y * SIZE + it] * buffer_b[it * SIZE + local_x];
        }
        __syncthreads();
    }

    const int x = block_x + local_x;
    const int y = block_y + local_y;
    if (x < w && y < h) {
        c[y * w + x] = buffer_c[local_index];
    }
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b, gpu::gpu_mem_32f& c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
