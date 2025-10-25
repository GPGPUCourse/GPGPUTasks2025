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
    // TODO

    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    float acc = 0.0f;
    for (int ki = 0; ki < k; ++ki) {
        acc += a[j * k + ki] * b[ki * w + i];
    }
    c[j * w + i] = acc;
}
