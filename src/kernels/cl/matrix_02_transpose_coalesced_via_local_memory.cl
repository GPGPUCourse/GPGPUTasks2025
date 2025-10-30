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
    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    const unsigned int global_i = get_global_id(0); // column in input
    const unsigned int global_j = get_global_id(1); // row in input
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    if (global_i < w && global_j < h) {
        tile[local_j][local_i] = matrix[global_j * w + global_i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int group_i = get_group_id(0);
    const unsigned int group_j = get_group_id(1);
    const unsigned int transposed_global_i = group_j * GROUP_SIZE_Y + local_i;
    const unsigned int transposed_global_j = group_i * GROUP_SIZE_X + local_j;

    if (transposed_global_i < h && transposed_global_j < w) {
        transposed_matrix[transposed_global_j * h + transposed_global_i] = tile[local_i][local_j];
    }
}
