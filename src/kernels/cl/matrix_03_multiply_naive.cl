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
    const unsigned int i = get_global_id(0); // column index in output C
    const unsigned int j = get_global_id(1); // row index in output C

    if (i >= w || j >= h) return;

    // C[j, i] = sum(A[j, ki] * B[ki, i] for ki in 0..k)
    float acc = 0.0f;
    for (unsigned int ki = 0; ki < k; ++ki) {
        acc += a[j * k + ki] * b[ki * w + i];
    }

    c[j * w + i] = acc;
}
