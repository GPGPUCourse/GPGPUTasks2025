#include "../defines.h"

#define TILE_SIZE GROUP_SIZE_X

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    const unsigned int global_column = get_global_id(0);
    const unsigned int global_row = get_global_id(1);

    const unsigned int local_column = get_local_id(0);
    const unsigned int local_row = get_local_id(1);

    __local float local_data_a[TILE_SIZE][TILE_SIZE];
    __local float local_data_b[TILE_SIZE][TILE_SIZE];

    float result = 0;

    for (unsigned int shift = 0; shift < k; shift += TILE_SIZE) {
        if (global_row < h && shift + local_column < k) {
            local_data_a[local_row][local_column] = a[global_row * k + shift + local_column];
        } else {
            local_data_a[local_row][local_column] = 0;
        }

        if (global_column < w && shift + local_row < k) {
            local_data_b[local_column][local_row] = b[(shift + local_row) * w + global_column];
        } else {
            local_data_b[local_column][local_row] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i) {
            result += local_data_a[local_row][i] * local_data_b[local_column][i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_column < w && global_row < h) {
        c[global_row * w + global_column] = result;
    }
}
