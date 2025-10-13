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
    unsigned int global_idx_x = get_global_id(0);
    unsigned int global_idx_y = get_global_id(1);

    if (global_idx_x >= w || global_idx_y >= h) {
        return;
    }
    float sum_result = 0;
    for (unsigned int i = 0; i < k; i++) {
        sum_result += a[global_idx_y * k + i] * b[i * w + global_idx_x];
    }
    c[global_idx_y * w + global_idx_x] = sum_result;
}
