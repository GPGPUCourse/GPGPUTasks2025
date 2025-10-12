#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // TODO
    __local float local_matrix[256];
    
    const uint global_index_x = get_global_id(0);
    const uint global_index_y = get_global_id(1);

    const uint local_index_x = get_local_id(0);
    const uint local_index_y = get_local_id(1);

    if (global_index_x < w && global_index_y < h) {
        const uint local_index = (local_index_x + local_index_y) % 16 + local_index_y * 16;
        const uint global_index = global_index_y * w + global_index_x;
        local_matrix[local_index] = matrix[global_index];    
        return;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_index_x < w && global_index_y < h) {
        const uint global_work_group_first_index_y = global_index_x - local_index_x; 
        const uint global_work_group_first_index_x = global_index_y - local_index_y; 

        const uint current_global_index_x = global_index_x + local_index_x;
        const uint current_global_index_y = global_index_y + local_index_y;

        const uint local_index = (local_index_x + local_index_y) % 16 + local_index_x * 16;
        const uint global_index = current_global_index_x * h + current_global_index_y;
        transposed_matrix[global_index] = local_matrix[local_index];
    }
}
