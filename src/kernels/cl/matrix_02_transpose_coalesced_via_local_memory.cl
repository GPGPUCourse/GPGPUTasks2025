#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,
                       __global       float* transposed_matrix,
                                uint w,
                                uint h)
{
    __local float loc_mem[GROUP_SIZE_Y][GROUP_SIZE_X];
    
    uint global_col = get_global_id(0);
    uint global_row = get_global_id(1);

    uint local_col = get_local_id(0);
    uint local_row = get_local_id(1);
    
    uint group_col = get_group_id(0);
    uint group_row = get_group_id(1);
    
    uint final_col = group_row * GROUP_SIZE_Y + local_col;
    uint final_row = group_col * GROUP_SIZE_X + local_row;

    uint idx;

    if (global_col < w && global_row < h) {
        idx = global_row * w + global_col;
        loc_mem[local_row][local_col] = matrix[idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (final_col < h && final_row < w) {
        idx = final_row * h + final_col;
        transposed_matrix[idx] = loc_mem[local_col][local_row];
    }
}
