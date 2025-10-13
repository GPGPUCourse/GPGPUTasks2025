#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_01_transpose_naive(
                       __global const float* matrix,            // w * h
                       __global       float* transposed_matrix, // h * w
                                unsigned int w,
                                unsigned int h) 
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    
    if (x < w && y < h) {

        float value = matrix[y * w + x];
        transposed_matrix[x * h + y] = value;
    }
}
