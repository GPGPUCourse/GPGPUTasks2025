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
    const unsigned int global_row = get_global_id(1);
    const unsigned int global_col = get_global_id(0);

    const unsigned int local_row = get_local_id(1);
    const unsigned int local_col = get_local_id(0);

    // read to local memory
    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X];
    if (global_row < h && global_col < w) {
        local_data[local_row][(local_col + local_row) % GROUP_SIZE_X] = matrix[global_row * w + global_col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // transpose
    unsigned int transposed_global_row = get_group_id(0) * GROUP_SIZE_Y + local_row;
    unsigned int transposed_global_col = get_group_id(1) * GROUP_SIZE_X + local_col;

    if (transposed_global_row >= w || transposed_global_col >= h) return;

    transposed_matrix[transposed_global_row * h + transposed_global_col] = local_data[local_col][(local_row + local_col) % GROUP_SIZE_Y];
}
