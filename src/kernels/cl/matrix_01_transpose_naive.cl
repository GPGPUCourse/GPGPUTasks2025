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
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);

    if (col >= w || row >= h) {
        return;
    }

    const uint index = row * w + col;
    const uint transposed_index = col * h + row;

    transposed_matrix[transposed_index] = matrix[index];
}
