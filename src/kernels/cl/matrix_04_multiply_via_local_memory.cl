#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_XY, GROUP_SIZE_XY, 1)))
__kernel void
matrix_04_multiply_via_local_memory(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    uint tile_x = x - lx;
    uint tile_y = y - ly;

    __local float a_local[GROUP_SIZE_XY][GROUP_SIZE_XY + 1];
    __local float b_local[GROUP_SIZE_XY][GROUP_SIZE_XY + 1];

    float val = 0;

    for (uint tile_k = 0; tile_k < k; tile_k += GROUP_SIZE_XY) {
        // load A tile(tile_k, tile_y)
        if (y < h && lx + tile_k < k)
            a_local[ly][lx] = a[y * k + (tile_k + lx)];
        else
            a_local[ly][lx] = 0;

        // load B tile(tile_x, tile_k)
        if (x < w && ly + tile_k < k)
            b_local[ly][lx] = b[(tile_k + ly) * w + x];
        else
            b_local[ly][lx] = 0;

        // both coalesced

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < GROUP_SIZE_XY; i++)
            val += a_local[ly][i] * b_local[i][lx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (y < h && x < w)
        c[y * w + x] = val;
}
