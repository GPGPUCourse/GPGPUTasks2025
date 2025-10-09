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
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (row >= h || col >= w) {
        return;
    }

    float res = 0;

    for (unsigned int i = 0; i < k; i++) {
        res += a[row * k + i] * b[col + w * i];
    }

    c[row * w + col] = res;
}
