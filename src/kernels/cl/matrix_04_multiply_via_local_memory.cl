#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    __local float tile_a[16][16];
    __local float tile_b[16][16];

    unsigned int local_x = get_local_id(0);
    unsigned int local_y = get_local_id(1);
    unsigned int global_x = get_global_id(0);
    unsigned int global_y = get_global_id(1);

    unsigned int size = (k + 15) / 16;
    float s = 0.0;
    for (unsigned int i = 0; i < size; i++) {
        unsigned int x = i * 16 + local_x;
        if (global_y >= h || x >= k) {
            tile_a[local_y][local_x] = 0.0f;
        } else {
            tile_a[local_y][local_x] = a[global_y * k + x];
        }
        unsigned int y = i * 16 + local_y;
        if (y >= k || global_x >= w) {
            tile_b[local_y][local_x] = 0.0f;
        } else {
            tile_b[local_y][local_x] = b[y * w + global_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int j = 0; j < 16; j++) {
            s += tile_a[local_y][j] * tile_b[j][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_x < w && global_y < h) {
        c[global_y * w + global_x] = s;
    }
}
