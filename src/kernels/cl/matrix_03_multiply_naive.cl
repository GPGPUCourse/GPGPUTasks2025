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
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    if (x >= w || y >= h)
        return;
    
    float res = 0.f;
    for (int p = 0; p < k; ++p)
        res += a[y * k + p] * b[p * w + x];
    
    c[y * w + x] = res;
}
