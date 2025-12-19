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
    const uint row = get_global_id(1);
    const uint col = get_global_id(0);

    if (row >= h || col >= w) return;

    float acc = 0;
    for (uint t = 0; t < k; ++t) {
        acc += a[row * k + t] * b[t * w + col];
    }
    c[row * w + col] = acc;
}
