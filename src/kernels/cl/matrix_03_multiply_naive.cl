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
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    if (h <= i || w <= j) {
        return;
    }

    float sum = 0;

    for (unsigned int l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * w + j];
    }

    c[i * w + j] = sum;
}
