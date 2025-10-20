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
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.f;
    for (int i = 0; i * TILE_SIZE < k; i++) {
        tile_a[ly][lx] = a[gy * k + lx + (i * TILE_SIZE)];
        tile_b[ly][lx] = b[(ly + i * TILE_SIZE) * w + gx];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tile_a[ly][j] * tile_b[j][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[gy * w + gx] = sum;
}
