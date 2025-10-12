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
    unsigned int index  = get_global_id(0);

    unsigned int curX = index%w;
    unsigned int curY = index/w;
    if (index< w*h){
    transposed_matrix[curX * h + curY] = matrix[index];
    }
    

}
