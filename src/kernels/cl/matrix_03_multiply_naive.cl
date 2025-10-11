#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_03_multiply_naive(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    const int global_row = get_global_id(1);
    const int global_col = get_global_id(0);
    if (global_row >= h || global_col >= w) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += a[global_row * k + i] * b[i * w + global_col];
    }
    c[global_row * w + global_col] = sum;
}
