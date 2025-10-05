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
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    
    if (x >= w || y >= h) {
        return;
    }
    
    float buffer = 0.0f;
    for (uint ki = 0; ki < k; ++ki) {
        buffer += a[y * k + ki] * b[ki * w + x];
    }
    c[y * w + x] = buffer;
}
