#ifdef CLANGD
#include <__clang_cuda_builtin_vars.h>
#endif
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
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..w
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; // 0..h

    if (i >= w) return;
    if (j >= h) return;
    curassert(GROUP_SIZE_X == blockDim.x, 714029210);
    curassert(GROUP_SIZE_Y == blockDim.y, 714029211);

    const uint i_local = threadIdx.x;
    const uint j_local = threadIdx.y;
     __shared__ float tile_a[GROUP_SIZE_X * GROUP_SIZE_Y];
     __shared__ float tile_b[GROUP_SIZE_Y * GROUP_SIZE_X];
    uint tile_a_ix = j_local * GROUP_SIZE_X + i_local;
    uint tile_b_ix = i_local * GROUP_SIZE_X + j_local; // transpose b on the way

    double acc = 0.0;
    const uint tile_cnt = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;
    for (uint ti = 0; ti < tile_cnt; ++ti) {
        uint a_row = j, a_col = (GROUP_SIZE_X * ti) + i_local;
        uint b_row = (GROUP_SIZE_X * ti) + j_local, b_col = i;
        if (a_col < k) {
            uint a_ix = a_row * k + a_col;
            tile_a[tile_a_ix] = a[a_ix];
        } else {
            tile_a[tile_a_ix] = 0;
        }
        if (b_row < k) {
            uint b_ix = b_row * w + b_col;
            tile_b[tile_b_ix] = b[b_ix];
        } else {
            tile_b[tile_b_ix] = 0;
        }

        // if (i == 16 && j == 0) printf("i,j=%d,%d, li,lj=%d,%d tile=%d a=%d,%d b=%d,%d\n", i,j, i_local,j_local, ti, a_col,a_row, b_col,b_row);
        // if (i == 16 && j == 0) printf("i,j=%d,%d, li,lj=%d,%d tile_a=%f tile_b=%f\n", i,j, i_local,j_local, tile_a[tile_b_ix], tile_b[tile_b_ix]);

        // {
        // __syncthreads();

        // if (blockIdx.x == 5 && blockIdx.y == 5 && i_local == 0 && j_local == 0) {
        //     printf("block %d,%d tile %d\n", blockIdx.x, blockIdx.y, ti);
        //     printf("a view\n");
        //     for (int y = j; y < j + GROUP_SIZE_Y; y++) {
        //         for (int x = GROUP_SIZE_X*ti; x < min(GROUP_SIZE_X*(ti+1), k); x++) {
        //             printf("%6.1f ", a[y*k + x]);
        //         }
        //         printf("\n");
        //     }
        //     printf("a tile view\n");
        //     for (int y = 0; y < GROUP_SIZE_Y; y++) {
        //         for (int x = 0; x < GROUP_SIZE_X; x++) {
        //             printf("%6.1f ", tile_a[y*GROUP_SIZE_X + x]);
        //         }
        //         printf("\n");
        //     }
        //     printf("b view\n");
        //     for (int y = GROUP_SIZE_X*ti; y < min(GROUP_SIZE_X*(ti+1), k); y++) {
        //         for (int x = i; x < i + GROUP_SIZE_X; x++) {
        //             printf("%6.1f ", b[y*w + x]);
        //         }
        //         printf("\n");
        //     }
        //     printf("b tile view\n");
        //     for (int y = 0; y < GROUP_SIZE_Y; y++) {
        //         for (int x = 0; x < GROUP_SIZE_X; x++) {
        //             printf("%6.1f ", tile_b[y*GROUP_SIZE_X + x]);
        //         }
        //         printf("\n");
        //     }
        // }
        // }
        __syncthreads();

        uint tile_a_row = j_local;
        uint tile_b_row = i_local;
        for (uint t = 0; t < GROUP_SIZE_X; ++t) {
            float a_val = tile_a[tile_a_row * GROUP_SIZE_X + t];
            float b_val = tile_b[tile_b_row * GROUP_SIZE_Y + t];
            acc += a_val * b_val;
        }
    }

    uint c_ix = j*w + i;
    c[c_ix] = acc;
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
