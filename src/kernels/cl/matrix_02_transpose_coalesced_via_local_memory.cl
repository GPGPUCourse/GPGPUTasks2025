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
    const uint index_x = get_local_id(0);
    const uint index_y = get_local_id(1);
    const uint group_x = get_group_id(0);
    const uint group_y = get_group_id(1);
    const uint global_x = get_global_id(0);
    const uint global_y = get_global_id(1);
    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X];

    if (global_x < w && global_y < h) {
        local_data[index_y][index_x] = matrix[global_x + global_y * w];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint transposed_x = group_y * GROUP_SIZE_Y + index_x;
    const uint transposed_y = group_x * GROUP_SIZE_X + index_y;

    if (transposed_x < h && transposed_y < w) {
        transposed_matrix[transposed_y * h + transposed_x] = local_data[index_x][index_y];
    }
}
