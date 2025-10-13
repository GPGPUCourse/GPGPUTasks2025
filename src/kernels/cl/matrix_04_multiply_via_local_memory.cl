#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

#define TILE_DIM 16

__attribute__((reqd_work_group_size(TILE_DIM, TILE_DIM, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float a_tile[TILE_DIM][TILE_DIM + 1];
    __local float b_tile[TILE_DIM][TILE_DIM + 1];

    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    const unsigned int local_j = get_local_id(0);
    const unsigned int local_i = get_local_id(1);

    const unsigned int iters = (k + TILE_DIM - 1) / TILE_DIM;

    float sum = 0;

    for (unsigned int block = 0; block < iters; ++block) {
        const unsigned int a_col = block * TILE_DIM + local_j;
        const unsigned int a_row = i;
        if (a_row < h && a_col < k) {
            a_tile[local_i][local_j] = a[a_row * k + a_col];
        } else {
            a_tile[local_i][local_j] = 0.0f;
        }

        const unsigned int b_col = j;
        const unsigned int b_row = block * TILE_DIM + local_i;
        if (b_row < k && b_col < w) {
            b_tile[local_i][local_j] = b[b_row * w + b_col];
        } else {
            b_tile[local_i][local_j] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int l = 0; l < TILE_DIM; ++l) {
            sum += a_tile[local_i][l] * b_tile[l][local_j];
        }

        // чтобы никто не поменял значение, пока считается сумма
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < h && j < w) {
        c[i * w + j] = sum;
    }
}
