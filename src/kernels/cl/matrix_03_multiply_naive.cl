#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{

    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);

    const uint global_x = get_group_id(0) * GROUP_SIZE_X + local_x;
    const uint global_y = get_group_id(1) * GROUP_SIZE_Y + local_y;

    if (global_x >= w || global_y >= h) {
        return;
    }

    float sum = 0.0f;

    for (uint i = 0; i < k; ++i) {
        float a_val = a[i + global_y * k];
        float b_val = b[global_x + i * w];
        sum += a_val * b_val;
    }

    c[global_x + global_y * w] = sum;
}
