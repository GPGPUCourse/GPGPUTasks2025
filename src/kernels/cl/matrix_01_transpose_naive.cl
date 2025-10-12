#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_01_transpose_naive(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // TODO
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= w || y >= h) return;

    const uint in_idx  = y * w + x;
    const uint out_idx = x * h + y;
    transposed_matrix[out_idx] = matrix[in_idx];
}
