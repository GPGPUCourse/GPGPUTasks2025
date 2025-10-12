#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const uint global_index_x = get_global_id(0);
    const uint global_index_y = get_global_id(1);

    if (global_index_x >= w || global_index_y >= h) {
        return;
    }

    float accumulated = 0;
    for (uint i = 0; i < k; ++i) {
        accumulated += a[global_index_y * k + i] * b[i * w + global_index_x];
    }

    c[global_index_y * w + global_index_x] = accumulated;
}
