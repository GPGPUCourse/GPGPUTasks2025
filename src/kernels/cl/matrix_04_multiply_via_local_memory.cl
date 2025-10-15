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
    const unsigned int loc_x = get_local_id(0);
    const unsigned int loc_y = get_local_id(1);
    const unsigned int col = get_group_id(0) * GROUP_SIZE_X + loc_x;
    const unsigned int row = get_group_id(1) * GROUP_SIZE_Y + loc_y;

    __local float loc_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float loc_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0.0f;
    for (unsigned int i = 0; i < k; i += GROUP_SIZE_X) {
        const unsigned int a_col = i + loc_x;
        const unsigned int b_row = i + loc_y;

        if (row >= h || a_col >= k) {
            loc_a[loc_y][loc_x] = 0.0f;
        } else {
            loc_a[loc_y][loc_x] = a[row * k + a_col];
        }

        if (b_row >= k || col >= w) {
            loc_b[loc_y][loc_x] = 0.0f;
        } else {
            loc_b[loc_y][loc_x] = b[b_row * w + col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int ii = 0; ii < GROUP_SIZE_X; ++ii) {
            sum += loc_a[loc_y][ii] * loc_b[ii][loc_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row >= h || col >= w) return;

    c[row * w + col] = sum;
}
