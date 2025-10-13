#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    size_t g_col = get_global_id(0);
    size_t g_row = get_global_id(1);
    size_t l_col = get_local_id(0);
    size_t l_row = get_local_id(1);

    size_t block_x = get_group_id(0);
    size_t block_y = get_group_id(1);

    float sum = 0;
    size_t t_len = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (size_t i = 0; i < k / TILE_SIZE; i++) {
        size_t t_col = i * TILE_SIZE + l_col;
        size_t t_row = i * TILE_SIZE + l_row;
        tile_a[l_row][l_col] = a[g_row * k + t_col];
        tile_b[l_row][l_col] = b[t_row * w + g_col];

        barrier(CLK_LOCAL_MEM_FENCE);
        for(size_t j = 0; j < TILE_SIZE; j++) {
            sum += tile_a[l_row][j] * tile_b[j][l_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[g_row * w + g_col] = sum;
}
