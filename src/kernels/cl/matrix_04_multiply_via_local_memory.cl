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
    const int i = get_global_id(1);
    const int j = get_global_id(0);

    const int local_i = get_local_id(1);
    const int local_j = get_local_id(0);

    __local float buffer_a[GROUP_SIZE_Y][GROUP_SIZE_X];
    __local float buffer_b[GROUP_SIZE_Y][GROUP_SIZE_X];

    float sum = 0.0f;
    for (int wg = 0; wg < (k + GROUP_SIZE_X - 1) / GROUP_SIZE_X; wg++) {
        const int a_row = i;
        const int a_col = wg * GROUP_SIZE_X + local_j;
        if (a_row < h && a_col < k) {
            buffer_a[local_i][local_j] = a[a_row * k + a_col];
        } else {
            buffer_a[local_i][local_j] = 0.0f;
        }

        const int b_row = wg * GROUP_SIZE_Y + local_i;
        const int b_col = j;
        if (b_row < k && b_col < w) {
            buffer_b[local_i][local_j] = b[b_row * w + b_col];
        } else {
            buffer_b[local_i][local_j] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int ki = 0; ki < GROUP_SIZE_X; ki++) {
            sum += buffer_a[local_i][ki] * buffer_b[ki][local_j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < h && j < w) {
        c[i * w + j] = sum;
    }
}
