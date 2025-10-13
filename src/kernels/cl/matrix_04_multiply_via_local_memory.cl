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
    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);

    unsigned int global_x = get_global_id(0);
    unsigned int global_y = get_global_id(1);

    __local float a_sub[TILE_SIZE][TILE_SIZE];
    __local float b_sub[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    unsigned int tiles_num = k / TILE_SIZE;

    for(unsigned int t = 0; t < tiles_num; ++t)
    {
        const unsigned int a_col = t * TILE_SIZE + local_x;
        const unsigned int a_row = global_y;

        const unsigned int b_col = global_x;
        const unsigned int b_row = t * TILE_SIZE + local_y;

        a_sub[local_y][local_x] = a[a_row * k + a_col];
        b_sub[local_y][local_x] = b[b_row * w + b_col];

        barrier(CLK_LOCAL_MEM_FENCE);

        for(unsigned int i = 0; i < TILE_SIZE; ++i)
        {
            acc += a_sub[local_y][i] * b_sub[i][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_x < w && global_y < h) {
        c[global_y * w + global_x] = acc;
    }
}
