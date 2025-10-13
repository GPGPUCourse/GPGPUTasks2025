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
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);
    
    if (i >= w || j >= h) {
        return;
    }
    
    float sum = 0.0f;
    for (int z = 0; z < k; ++z) {
        sum += a[j * k + z] * b[z * w + i];
    }
    c[j * w + i] = sum;
}
