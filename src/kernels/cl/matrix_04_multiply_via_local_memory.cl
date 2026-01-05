#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

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
    const size_t global_row = get_global_id(0);
    const size_t global_column = get_global_id(1);

    const size_t local_row = get_local_id(0);
    const size_t local_column = get_local_id(1);

    const size_t group_row = get_group_id(0);
    const size_t group_column = get_group_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float part_sum = 0.0f;

    size_t tiles_num = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (size_t t = 0; t < tiles_num; ++t) {
        size_t a_row = local_row + group_row * TILE_SIZE;
        size_t a_column = local_column + t * TILE_SIZE;

        if (a_row < h && a_column < k) {
            tile_a[local_row][local_column] = a[a_row * k + a_column];
        }

        size_t b_row = local_row + t * TILE_SIZE;
        size_t b_column = local_column + group_column * TILE_SIZE;

        if (b_row < k && b_column < w) {
            tile_b[local_column][local_row] = b[b_row * w + b_column];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t i = 0; i < TILE_SIZE; ++i) {
            part_sum += tile_a[local_row][i] * tile_b[local_column][i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < h && global_column < w) {
        c[global_row * w + global_column] = part_sum;
    }
}
