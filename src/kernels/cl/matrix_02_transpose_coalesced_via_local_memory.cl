#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // w x h
    __global float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    const unsigned int global_column = get_global_id(0);
    const unsigned int global_row = get_global_id(1);

    const unsigned int local_column = get_local_id(0);
    const unsigned int local_row = get_local_id(1);

    const unsigned int group_column = get_group_id(0);
    const unsigned int group_row = get_group_id(1);

    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X];

    if (global_column < w && global_row < h) {
        local_data[local_row][(local_column + local_row) % GROUP_SIZE_X] = matrix[global_row * w + global_column];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int transposed_global_column = group_row * GROUP_SIZE_X + local_column;
    unsigned int transposed_global_row = group_column * GROUP_SIZE_Y + local_row;

    if (transposed_global_column < h && transposed_global_row < w) {
        transposed_matrix[transposed_global_row * h + transposed_global_column] = local_data[local_column][(local_row + local_column) % GROUP_SIZE_Y];
    }
}
