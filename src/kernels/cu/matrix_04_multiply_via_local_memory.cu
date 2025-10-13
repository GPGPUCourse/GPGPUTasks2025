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
    unsigned int idx_x = blockIdx.x * GROUP_SIZE_X + threadIdx.x; 
    unsigned int idx_y = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;

    if (idx_x >= w || idx_y >= h) return;
    
    const unsigned int tile_sz = GROUP_SIZE_X;

    __shared__ float a_tile[tile_sz][tile_sz];
    __shared__ float b_tile[tile_sz][tile_sz];  
    
    const unsigned int tiles_k = k  / tile_sz;
    float res = 0.0;

    for (int i = 0; i < tiles_k; ++i) {
        unsigned int col_idx = i * tile_sz + threadIdx.x; 
        unsigned int row_idx = i * tile_sz + threadIdx.y; 

        a_tile[threadIdx.y][threadIdx.x] = a[idx_y * k + col_idx];

        b_tile[threadIdx.y][threadIdx.x] = b[row_idx * w + idx_x];

        __syncthreads();

        for (int j = 0; j < tile_sz; ++j) {
            res += a_tile[threadIdx.y][j] * b_tile[j][threadIdx.x];
        }

        __syncthreads();
    }

    c[idx_y * w + idx_x] = res;
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
