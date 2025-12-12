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
    const uint col = get_global_id(0);
    const uint row = get_global_id(1);

    __local float local_data[GROUP_SIZE + 1];

    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint local_w = get_local_size(0);
    const uint local_h = get_local_size(1);

    const uint group_col = get_group_id(0);
    const uint group_row = get_group_id(1);

    if (row < h && col < w) {
        const uint index_read = local_row * local_w + local_col;
        local_data[index_read] = matrix[row * w + col];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (row < h && col < w) {
        const uint write_row = group_col * local_w + local_row;
        const uint write_col = group_row * local_h + local_col;
        const uint index_write = local_col * local_h + local_row;
    
        transposed_matrix[write_row * h + write_col] = local_data[index_write];
    }
}
