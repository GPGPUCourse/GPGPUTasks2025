#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

// because [GROUP_SIZE_X][GROUP_SIZE_X] looks strange
#define TILE_SIZE GROUP_SIZE_X

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    __local float A_local[GROUP_SIZE_Y][TILE_SIZE];
    __local float B_local[TILE_SIZE][GROUP_SIZE_X];

    float sum = 0.0f;

    for (unsigned block_k = 0; block_k < k; block_k += TILE_SIZE) {

        if (row < h && (block_k + local_col) < k) {
            A_local[local_row][local_col] = a[row * k + (block_k + local_col)];
        } else {
            A_local[local_row][local_col] = 0.0f;
        }

        if ((block_k + local_row) < k && col < w) {
            B_local[local_row][local_col] = b[(block_k + local_row) * w + col];
        } else {
            B_local[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned tile_k = 0; tile_k < TILE_SIZE; ++tile_k) {
            sum += A_local[local_row][tile_k] * B_local[tile_k][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < h && col < w) {
        c[row * w + col] = sum;
    }
}
