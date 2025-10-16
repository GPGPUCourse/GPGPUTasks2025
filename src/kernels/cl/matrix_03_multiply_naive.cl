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
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);
    if (i < w && j < h) {
        float sum = 0;
        for (size_t l = 0; l < k; ++l) {
            sum += a[j * k + l] * b[l * w + i];
        }
        c[j * w + i] = sum;
    }
}
