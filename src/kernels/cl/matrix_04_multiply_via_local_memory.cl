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

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int global_i = get_global_id(0); // column in C
    const unsigned int global_j = get_global_id(1); // row in C

    float acc = 0.0f;

    const unsigned int num_tiles = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    for (unsigned int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const unsigned int a_col = tile_idx * GROUP_SIZE_X + local_i;
        if (global_j < h && a_col < k) {
            tile_a[local_j][local_i] = a[global_j * k + a_col];
        } else {
            tile_a[local_j][local_i] = 0.0f;
        }

        const unsigned int b_row = tile_idx * GROUP_SIZE_X + local_j;
        if (b_row < k && global_i < w) {
            tile_b[local_j][local_i] = b[b_row * w + global_i];
        } else {
            tile_b[local_j][local_i] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int ki = 0; ki < GROUP_SIZE_X; ++ki) {
            acc += tile_a[local_j][ki] * tile_b[ki][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_i < w && global_j < h) {
        c[global_j * w + global_i] = acc;
    }
}
