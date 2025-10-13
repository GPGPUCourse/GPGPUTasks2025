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

    /*if (blockDim.x != blockDim.y) {
        printf("kernel requires blockDim.x == blockDim.y\n");
        return;
    }*/

    const unsigned int TILE_LEN = blockDim.x;

    const unsigned int glob_col = blockIdx.x * TILE_LEN + threadIdx.x;
    const unsigned int glob_row = blockIdx.y * TILE_LEN + threadIdx.y;

    if (glob_col >= w || glob_row >= h) {
        return;
    }

    const unsigned int glob_index = glob_row * w + glob_col;

    float accum = 0.;

    for (int tile_num = 0; tile_num < (k + TILE_LEN - 1) / TILE_LEN; ++tile_num)
    {
        __shared__ float workgroup_data_a[GROUP_SIZE_Y][GROUP_SIZE_X];

        const unsigned int a_col = tile_num * TILE_LEN + threadIdx.x;
        const unsigned int a_index = glob_row * k + a_col;

        if (a_col < k && glob_row < h) {
            workgroup_data_a[threadIdx.y][threadIdx.x] = a[a_index];
        } else {
            workgroup_data_a[threadIdx.y][threadIdx.x] = 0.;
        }

        __shared__ float workgroup_data_b[GROUP_SIZE_Y][GROUP_SIZE_X];

        const unsigned int b_row = tile_num * TILE_LEN + threadIdx.y;
        const unsigned int b_index = b_row * w + glob_col;
        
        if (b_row < k && glob_col < w) {
            workgroup_data_b[threadIdx.y][threadIdx.x] = b[b_index];
        } else {
            workgroup_data_b[threadIdx.y][threadIdx.x] = 0.;
        }   

        __syncthreads();


        for (int i = 0; i < TILE_LEN; ++i) {
            accum += workgroup_data_a[threadIdx.y][i] * workgroup_data_b[i][threadIdx.x];
        }

        __syncthreads();
    }

    c[glob_index] = accum;
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
