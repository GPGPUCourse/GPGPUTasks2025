#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_01_transpose_naive(
    __global const float* matrix, // w x h
    __global float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{

    const unsigned int global_idx_x = get_global_id(0);
    const unsigned int global_idx_y = get_global_id(1);

    if (global_idx_x >= w || global_idx_y >= h) {
        return;
    }
    // coalesced read, non-coalesced write
    transposed_matrix[global_idx_x * h + global_idx_y] = matrix[global_idx_y * w + global_idx_x];
}
