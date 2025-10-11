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
    const unsigned int row = get_global_id(1);
    const unsigned int col = get_global_id(0);

    if (col >= w || row >= h) return;

    float result = 0.0f;

    for (unsigned int pos = 0; pos < k; pos++) {
        result += a[row * k + pos] * b[pos * w + col];
    }

    c[row * w + col] = result;
}
