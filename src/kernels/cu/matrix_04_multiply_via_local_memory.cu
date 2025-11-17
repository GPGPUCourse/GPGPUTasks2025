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
    static_assert(GROUP_SIZE_X==GROUP_SIZE_Y);

    const uint index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint index_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (index_x >= w || index_y >= h)
        return;

    float result = 0;

    constexpr uint GROUP_SIZE_K = GROUP_SIZE_Y;
    __shared__ float local_data1[(GROUP_SIZE_K+1)*GROUP_SIZE_K];
    __shared__ float local_data2[(GROUP_SIZE_K+1)*GROUP_SIZE_K];
    const uint local_index_to = threadIdx.y * (GROUP_SIZE_K+1) + threadIdx.x;

    for (uint i = 0; i < k; i += GROUP_SIZE_K) {
        if (i + threadIdx.x < k)
            local_data1[local_index_to] = a[index_y * k + (i + threadIdx.x)];
        else
            local_data1[local_index_to] = 0;
        if (i + threadIdx.y < k)
            local_data2[local_index_to] = b[(i + threadIdx.y) * w + index_x];
        else
            local_data2[local_index_to] = 0;

        __syncthreads();

        for (uint j = 0; j < GROUP_SIZE_K; j++) {
            result += local_data1[threadIdx.y * (GROUP_SIZE_K+1) + j] * local_data2[j * (GROUP_SIZE_K+1) + threadIdx.x];
        }

        __syncthreads();
    }

    c[index_y * w + index_x] = result;
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
