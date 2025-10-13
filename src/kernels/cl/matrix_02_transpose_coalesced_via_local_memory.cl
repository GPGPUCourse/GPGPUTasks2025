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
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);

    if (i < w && j < h) {
        tile[local_j][local_i] = matrix[j * w + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint target_j = get_group_id(0) * GROUP_SIZE_Y + local_j;
    const uint target_i = get_group_id(1) * GROUP_SIZE_X + local_i;
    if (target_i < h && target_j < w) {
        transposed_matrix[target_j * h + target_i] = tile[local_i][local_j];
    }
}
