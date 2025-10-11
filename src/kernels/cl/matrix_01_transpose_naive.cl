#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__kernel void matrix_01_transpose_naive(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    const float data = matrix[j * w + i];
    transposed_matrix[i * h + j] = data;
}
