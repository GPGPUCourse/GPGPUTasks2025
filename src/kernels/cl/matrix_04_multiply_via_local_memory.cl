#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                uint w,
                                uint h,
                                uint k)
{
    __local float line_a[GROUP_SIZE_Y][GROUP_SIZE_X], line_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    uint global_col = get_global_id(0);
    uint global_row = get_global_id(1);

    uint local_col = get_local_id(0);
    uint local_row = get_local_id(1);
    uint n = (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X;

    float s = 0.0f;
    for (uint idx = 0; idx < n; ++idx) {
        uint a_col = idx * GROUP_SIZE_X + local_col;
        uint b_row = idx * GROUP_SIZE_X + local_row;
        if (global_row < h && a_col < k) {
            line_a[local_row][local_col] = a[global_row * k + a_col];
        }
        if (b_row < k && global_col < w) {
            line_b[local_row][local_col] = b[b_row * w + global_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < GROUP_SIZE_X; ++i) {
            s += line_a[local_row][i] * line_b[i][local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_col < w && global_row < h) {
        c[global_row * w + global_col] = s;
    }
}
