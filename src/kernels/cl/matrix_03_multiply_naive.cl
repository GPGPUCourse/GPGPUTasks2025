#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_03_multiply_naive(
    __global const float* a, // rows=h x cols=k
    __global const float* b, // rows=k x cols=w
    __global float* c, // rows=h x cols=w
    unsigned int w,
    unsigned int h,
    unsigned int k)
{
    int i = get_global_id(1);
    int j = get_global_id(0);
    float acc = 0.0f;

    for (int l = 0; l < k; ++l) {
        acc += a[i * k + l] * b[l * w + j];
    }
    c[i * w + j] = acc;
}
