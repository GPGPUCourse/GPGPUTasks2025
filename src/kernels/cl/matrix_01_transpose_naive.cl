#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_01_transpose_naive(
    __global const float *matrix,      // h x w
    __global float *transposed_matrix, // w x h
    unsigned int w,
    unsigned int h)
{
    const unsigned int height = get_global_id(0);
    const unsigned int width = get_global_id(1);
    if (width < w && height < h)
    {
        transposed_matrix[width * h + height] = matrix[height * w + width];
    }
}
