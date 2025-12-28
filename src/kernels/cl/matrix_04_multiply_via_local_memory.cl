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
    __local float A_tile[TILE_SIZE][TILE_SIZE];
    // Prevent bank conflict
    __local float B_tile[TILE_SIZE][TILE_SIZE + 1];

    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);

    size_t x = get_group_id(0) * TILE_SIZE + lx;
    size_t y = get_group_id(1) * TILE_SIZE + ly;

    float sum = 0.0f;

    for (uint ki = 0; ki < k; ki+=TILE_SIZE) {

        if (y < h && ki + lx < k) {
            A_tile[ly][lx] = a[y * k + ki + lx];
        } else {
            A_tile[ly][lx] = 0.0f;
        }

        if (x < w && ki + ly < k) {
            B_tile[ly][lx] = b[(ki + ly) * w + x];
        } else {
            B_tile[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < TILE_SIZE; ++i) {
            sum += A_tile[ly][i] * B_tile[i][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }  

    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
