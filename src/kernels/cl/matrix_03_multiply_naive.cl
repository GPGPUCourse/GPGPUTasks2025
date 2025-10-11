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
    unsigned int li = get_global_id(0);
    unsigned int lj = get_global_id(1);
    float sum = 0;

    for (unsigned int i = 0; i < k; i++) {
        sum += a[lj * k + i] * b[i * w + li];
    }

    c[lj * w + li] = sum;
}
