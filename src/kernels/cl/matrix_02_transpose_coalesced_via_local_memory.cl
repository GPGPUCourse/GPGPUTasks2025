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
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    
    const unsigned int tile_pitch = GROUP_SIZE_X + 1;
    
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);
    
    local float local_matrix[GROUP_SIZE_Y * tile_pitch];
    if (x < w && y < h) {
        local_matrix[local_y * tile_pitch + local_x] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    const unsigned int out_x = group_x * GROUP_SIZE_Y + local_y;
    const unsigned int out_y = group_y * GROUP_SIZE_X + local_x;
    if (out_x < h && out_y < w) {
        transposed_matrix[out_x * h + out_y] = local_matrix[local_x * tile_pitch + local_y];
    }
}
