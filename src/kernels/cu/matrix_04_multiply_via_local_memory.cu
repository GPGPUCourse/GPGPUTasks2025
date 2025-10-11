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
    unsigned int x = blockIdx.x * GROUP_SIZE_X + threadIdx.x; 
    unsigned int y = blockIdx.y * GROUP_SIZE_Y + threadIdx.y;
    
    __shared__ float tile_A[GROUP_SIZE_Y][GROUP_SIZE_X];
    __shared__ float tile_B[GROUP_SIZE_Y][GROUP_SIZE_X];  

    float tmp = 0.0f;
    if (x >= w || y >= h) return;
    const unsigned int numTiles = k  / GROUP_SIZE_X;

    for (unsigned int t = 0; t < numTiles; ++t) {
        unsigned int kx = t * GROUP_SIZE_X + threadIdx.x; 
        unsigned int ky = t * GROUP_SIZE_X + threadIdx.y; 
 
        tile_A[threadIdx.y][threadIdx.x] = a[y * k + kx];

        tile_B[threadIdx.y][threadIdx.x] = b[ky * w + x];

        __syncthreads();

        for (int j = 0; j < GROUP_SIZE_X; ++j) {
            tmp += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];
        }

        __syncthreads();
    }

    c[y * w + x] = tmp;
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
