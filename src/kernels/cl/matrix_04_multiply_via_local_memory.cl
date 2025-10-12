#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

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
    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint global_col = get_group_id(0) * TILE_SIZE + local_col;
    const uint global_row = get_group_id(1) * TILE_SIZE + local_row;

    __local float local_a[TILE_SIZE][TILE_SIZE], local_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    const uint count_tiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (uint tile = 0; tile < count_tiles; ++tile) {

        uint col_a = tile * TILE_SIZE + local_col;
        uint row_b = tile * TILE_SIZE + local_row;

        if (global_row < h && col_a < k) {
            local_a[local_row][local_col] = a[global_row * k + col_a];
        } else {
            local_a[local_row][local_col] = 0.0f;
        }

        if (global_col < w && row_b < k) {
            local_b[local_row][local_col] = b[row_b * w + global_col];
        } else {
            local_b[local_row][local_col] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += local_a[local_row][i] * local_b[i][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < h && global_col < w) {
        c[global_row * w + global_col] = sum;
    }
}
