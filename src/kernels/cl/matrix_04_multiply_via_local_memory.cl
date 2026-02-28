#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(__global const float* a, 
                                                  __global const float* b, 
                                                  __global       float* c, 
                                                  unsigned int w, unsigned int h, unsigned int k)
{
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);
    unsigned int gx = get_global_id(0);
    unsigned int gy = get_global_id(1);
    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0.0f;
    unsigned int num_tiles = k / GROUP_SIZE_X;

    for (unsigned int t = 0; t < num_tiles; ++t) {
        tile_a[ly][lx] = a[gy * k + (t * GROUP_SIZE_X + lx)];
        tile_b[ly][lx] = b[(t * GROUP_SIZE_Y + ly) * w + gx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int i = 0; i < GROUP_SIZE_X; ++i) {
            sum += tile_a[ly][i] * tile_b[i][lx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (gx < w && gy < h) {
        c[gy * w + gx] = sum;
    }
}