#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_03_multiply_naive(__global const float* a, 
                                       __global const float* b, 
                                       __global       float* c, 
                                       unsigned int w, unsigned int h, unsigned int k)
{
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    if (x < w && y < h) {
        float sum = 0.0f;
        for (unsigned int i = 0; i < k; ++i) {
            sum += a[y * k + i] * b[i * w + x];
        }
        c[y * w + x] = sum;
    }
}