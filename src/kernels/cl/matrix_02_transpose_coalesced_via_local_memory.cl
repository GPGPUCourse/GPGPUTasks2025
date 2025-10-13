#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h (h rows, w cols)
                       __global       float* transposed_matrix, // h x w (w rows, h cols)
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int j = get_global_id(0);
    const unsigned int i = get_global_id(1);

    const unsigned int local_j = get_local_id(0);
    const unsigned int local_i = get_local_id(1);

    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    if (i < h && j < w) {
        local_data[local_i][local_j] = matrix[i * w + j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int dest_group_i = get_group_id(0);
    const unsigned int dest_group_j = get_group_id(1);

    const unsigned int dest_i = dest_group_i * GROUP_SIZE_X + local_i;
    const unsigned int dest_j = dest_group_j * GROUP_SIZE_Y + local_j;

    if (dest_i < w && dest_j < h) {
        transposed_matrix[dest_i * h + dest_j] = local_data[local_j][local_i];
    }
}
