#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X + 1];


    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint i = get_local_id(0);
    uint j = get_local_id(1);

    local_data[j][i] = matrix[y * w + x];

    uint new_x = (y - j) + i;
    uint new_y = (x - i) + j;

    barrier(CLK_LOCAL_MEM_FENCE);

    transposed_matrix[new_y * h + new_x] = local_data[i][j];
}
