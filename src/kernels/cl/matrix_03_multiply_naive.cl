#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_03_multiply_naive(
    __global const float *a, // rows=h x cols=k
    __global const float *b, // rows=k x cols=w
    __global float *c,       // rows=h x cols=w
    unsigned int h,
    unsigned int w,
    unsigned int k)
{
    const uint height = get_global_id(0);
    const uint width = get_global_id(1);
    if (height < h && width < w)
    {
        float sum = 0;
        for (uint i = 0; i < k; i++)
        {
            sum += a[height * k + i] * b[i * w + width];
        }
        c[height * w + width] = sum;
    }
}
