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
    const unsigned int x = get_global_id(0); // w
    const unsigned int y = get_global_id(1); // h

    const unsigned int local_x = get_local_id(0); // w
    const unsigned int local_y = get_local_id(1); // h

    __local float tile_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float tile_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    float matmul = 0.0f;

    const unsigned int tiles_per_row = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;
    for (int i = 0; i < tiles_per_row; ++i) {
        if (y < h && (i * GROUP_SIZE_X + local_x) < k) {
            tile_a[local_y][local_x] = a[y * k + (i * GROUP_SIZE_X + local_x)];
        } else {
            tile_a[local_y][local_x] = 0.0f;
        }
        if ((i * GROUP_SIZE_Y + local_y) < k && x < w) {
            tile_b[local_y][local_x] = b[(i * GROUP_SIZE_X + local_y) * w + x];
        } else {
            tile_b[local_y][local_x] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < GROUP_SIZE_X; ++j) {
            matmul += tile_a[local_y][j] * tile_b[j][local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < w && y < h) {
        c[y * w + x] = matmul;
    }
}
