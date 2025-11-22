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
    int j = get_global_id(0);
    int i = get_global_id(1);
    float sum_el = 0;
    if (i >= h || j >= w) {
        return;
    }
    for (int tmp = 0; tmp < k; ++tmp) {
        sum_el += a[i * k + tmp] * b[j + w * tmp];
    }
    c[i * w + j] = sum_el;
}
