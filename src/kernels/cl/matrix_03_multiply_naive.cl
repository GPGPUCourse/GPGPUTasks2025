#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);

    if (global_x >= w || global_y >= h) {
        return;
    }

    float acc = 0;
    for (unsigned int i = 0; i < k; ++i) {
        acc += a[i + k * global_y] * b[i * w + global_x];
    }

    c[global_x + w * global_y] = acc;
}
