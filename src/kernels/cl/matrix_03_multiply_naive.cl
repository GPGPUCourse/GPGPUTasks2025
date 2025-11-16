#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                uint w,
                                uint h,
                                uint k)
{
    uint col = get_global_id(0);
    uint row = get_global_id(1);
    
    if (col < w || row < h) {
        float s = 0.0f;
        for (uint i = 0; i < k; ++i) {
            s += a[row * k + i] * b[i * w + col];
        }
        c[row * w + col] = s;
    }
}
