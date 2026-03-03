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
    const uint loc_i = get_local_id(0);
    const uint loc_j = get_local_id(1);
    const uint gr_i = get_group_id(0);
    const uint gr_j = get_group_id(1);
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    __local float tile[GROUP_SIZE_X + 1][GROUP_SIZE_Y];

    if (i < w && j < h) {
        tile[loc_i][loc_j] = matrix[j * w + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint new_gr_i = gr_j;
    const uint new_gr_j = gr_i;
    const uint new_i = new_gr_i * GROUP_SIZE_Y + loc_i;
    const uint new_j = new_gr_j * GROUP_SIZE_X + loc_j;

    if (new_i < h && new_j < w) {
        transposed_matrix[new_j * h + new_i] = tile[loc_j][loc_i];
    }
}
