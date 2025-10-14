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
    const uint index_x = get_global_id(0);
    const uint index_y = get_global_id(1);
    c[index_x + w * index_y] = 0;
    for (int i = 0; i < k; ++i) {
        c[index_x + w * index_y] += a[i + index_y * k] * b[index_x + i * w];
    }
}
