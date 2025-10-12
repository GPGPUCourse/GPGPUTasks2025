#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void matrix_01_transpose_naive(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    unsigned int row = get_global_id(1);
    unsigned int col = get_global_id(0);

    if (row >= h || col >= w)
        return;

    transposed_matrix[col * h + row] = matrix[row * w + col];
}
