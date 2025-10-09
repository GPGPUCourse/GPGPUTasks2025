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
    const unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int local_x = threadIdx.x;
    const unsigned int local_y = threadIdx.y;

    __shared__ float matrix_a_gpu[GROUP_SIZE];
    __shared__ float matrix_b_gpu[GROUP_SIZE];
    __shared__ float matrix_res_gpu[GROUP_SIZE];

    c[index_y * w + index_x] = 0.f;
    const unsigned int res_ind = local_y * blockDim.x + local_x;
    matrix_res_gpu[res_ind] = 0.f;

    for (unsigned int i = 0; i < k; i += blockDim.x) {
        const unsigned int x_a = i + local_x;
        const unsigned int y_a = index_y;
        matrix_a_gpu[res_ind] = a[y_a * k + x_a];

        const unsigned int x_b = index_x;
        const unsigned int y_b = i + local_y;
        matrix_b_gpu[res_ind] = b[y_b * w + x_b];

        __syncthreads();
        for (unsigned int j = 0; j < blockDim.x; j++) {
            const unsigned int a_ind = local_y * blockDim.x + j;
            const unsigned int b_ind = j * blockDim.x + local_x;

            matrix_res_gpu[res_ind] += matrix_a_gpu[a_ind] * matrix_b_gpu[b_ind];
        }
        __syncthreads();
    }

    c[index_y * w + index_x] = matrix_res_gpu[res_ind];
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
