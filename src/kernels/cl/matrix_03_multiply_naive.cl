#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(32, 32, 1)))
__kernel void matrix_03_multiply_naive(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{
    const unsigned int globalX = get_global_id(0);
    const unsigned int globalY = get_global_id(1);

    if (globalX >= w || globalY >= h) {
        return;
    }

    float sum = 0;

    for (unsigned int i = 0; i != k; ++i) {
        sum += a[globalY * k + i] * b[i * w + globalX];
    }

    c[globalY * w + globalX] = sum;
}
