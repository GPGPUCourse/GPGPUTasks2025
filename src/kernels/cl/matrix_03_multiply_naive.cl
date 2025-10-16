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
    // TODO
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= w || y >= h) return;

    float acc = 0.0f;
    const uint a_row = y * k;
    for (uint kk = 0; kk < k; ++kk) {
        acc += a[a_row + kk] * b[kk * w + x];
    }
    c[y * w + x] = acc;
}
