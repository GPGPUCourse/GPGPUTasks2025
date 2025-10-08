#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    // TODO
    uint i = get_global_id(0);
    uint j = get_global_id(1);

    if (i >= w || j >= h) return;

    float result = 0;

    for (uint l = 0; l < k; l++) {
        result = fma (a[j * k + l], b[l * w + i], result);
    }

    c [j * w + i] = result;

}
