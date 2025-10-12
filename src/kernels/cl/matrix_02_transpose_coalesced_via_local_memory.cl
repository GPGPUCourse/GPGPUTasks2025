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
    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);
    const uint group_x = get_group_id(0);
    const uint group_y = get_group_id(1);

    __local float local_data[GROUP_SIZE_X][GROUP_SIZE_Y];

    uint global_1d_x = group_x * GROUP_SIZE + local_x;
    uint global_1d_y = group_y * GROUP_SIZE + local_y;

    if (global_1d_x >= w || global_1d_y >= h)  {
        return;
    }

    local_data[local_x][local_y] = matrix[global_1d_y * w + global_1d_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    uint transposed_x = group_y * GROUP_SIZE + local_x;
    uint transposed_y = group_x * GROUP_SIZE + local_y;

    if (transposed_x >= h || transposed_y >= w) {
        return;
    }

    transposed_matrix[transposed_y * h + transposed_x] = local_data[local_y][local_x];
}
