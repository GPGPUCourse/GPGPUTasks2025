#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_01_transpose_naive(
    __global const float* matrix, // rows=h x cols=w
    __global float* transposed_matrix, // rows=w x cols=h
    unsigned int w,
    unsigned int h)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if (i >= w || j >= h) {
        return;
    }

    transposed_matrix[j + h * i] = matrix[i + w * j];
}
