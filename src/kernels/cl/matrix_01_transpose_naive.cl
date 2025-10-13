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
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);
    const unsigned int index_m = i * w + j;
    const unsigned int index_tm = j * h + i;
    if (i < h && j < w) {
        transposed_matrix[index_tm] = matrix[index_m];
    }
}
