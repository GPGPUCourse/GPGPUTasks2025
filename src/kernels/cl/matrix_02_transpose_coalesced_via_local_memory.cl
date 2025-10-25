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
    const uint global_i = get_global_id(0);
    const uint global_j = get_global_id(1);

    __local float local_data[GROUP_SIZE_X][GROUP_SIZE_Y];

    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);

    local_data[local_j][local_i] = matrix[global_j * w + global_i];

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint transposed_i = get_group_id(1) * GROUP_SIZE_Y + local_i;
    const uint transposed_j = get_group_id(0) * GROUP_SIZE_X + local_j;

    transposed_matrix[transposed_j * h + transposed_i] = local_data[local_i][local_j];
}
