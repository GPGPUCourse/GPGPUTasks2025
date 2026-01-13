#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_03_multiply_naive(
    __global const float* a, // h x k
    __global const float* b, // k x w
    __global       float* c, // h x w
    const uint w,
    const uint h,
    const uint k)
{
    const uint x = get_global_id(0); // column
    const uint y = get_global_id(1); // row

    if (x < w && y < h) {
        float sum = 0.0f;

        for (uint i = 0; i < k; ++i) {
            sum += a[y * k + i] * b[i * w + x];
        }

        c[y * w + x] = sum;
    }
}

