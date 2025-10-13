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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);

    __local float a_tile[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float b_tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0;
    const uint number_of_tiles = (GROUP_SIZE_X + k - 1) / GROUP_SIZE_X;

    for (uint i = 0; i < number_of_tiles; i++) {
        const uint a_col = i * GROUP_SIZE_X + local_x;
        const uint b_row = i * GROUP_SIZE_Y + local_y;

        a_tile[local_y][local_x] = (y < h && a_col < k) ? a[y * k + a_col] : 0;
        b_tile[local_y][local_x] = (x < w && b_row < k) ? b[b_row * w + x] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint j = 0; j < GROUP_SIZE_X; j++) {
            sum += a_tile[local_y][j] * b_tile[j][local_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (x < w && y < h) {
        c[y * w + x] = sum;
    }
}
