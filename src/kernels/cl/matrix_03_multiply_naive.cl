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
    uint i = get_global_id(1);
    uint j = get_global_id(0);

    float res = 0;
    for (int m=0; m<k; ++m) {
        res += a[i * k + m] * b[m * w + j];
    }
    c[i * w + j] = res;
}
