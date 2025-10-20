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
    size_t gx = get_global_id(0); // in [0, w)
    size_t gy = get_global_id(1); // in [0, h)

    float sum = 0.0;
    for (int j = 0; j < k; ++j) {
        sum += a[gy * k + j] * b[j * w + gx];
    }
    c[gy * w + gx] = sum;
}
