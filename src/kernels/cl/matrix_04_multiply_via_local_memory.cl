#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    float sum = 0.0f;

    const unsigned int num_tiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (unsigned int t = 0; t < num_tiles; ++t) {
        const unsigned int tile_col = t * GROUP_SIZE_X + lx;
        const unsigned int tile_row = t * GROUP_SIZE_Y + ly;

        if (row < h && tile_col < k) {
            tile_a[ly][lx] = a[row * k + tile_col];
        } else {
            tile_a[ly][lx] = 0.0f;
        }

        if (tile_row < k && col < w) {
            tile_b[ly][lx] = b[tile_row * w + col];
        } else {
            tile_b[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i) {
            sum += tile_a[ly][i] * tile_b[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < h && col < w) {
        c[row * w + col] = sum;
    }
}
