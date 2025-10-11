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
    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];

    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    if (global_x < w && global_y < h) {
        // NO BANK CONFLICTS! YAY!
        local_data[(local_x + local_y) % GROUP_SIZE_X + GROUP_SIZE_X * local_y] = matrix[global_x + w * global_y];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_x < w && global_y < h) {
        transposed_matrix[global_x * h + global_y] = local_data[(local_x + local_y) % GROUP_SIZE_X + GROUP_SIZE_X * local_y];
    }
}
