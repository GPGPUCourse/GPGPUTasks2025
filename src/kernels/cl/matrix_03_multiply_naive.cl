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
    const unsigned int index_w = get_global_id(0);
    const unsigned int index_h = get_global_id(1);

    if (index_w >= w || index_h >= h) {
        return;
    }

    float summ = 0.0f;

    for (unsigned int i = 0; i < k; i++) {
        summ += a[index_h * k + i] * b[index_w + i * w];
    }

    c[index_w + w * index_h] = summ;
}
