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
    __local float local_data[GROUP_SIZE];

    const uint col = get_global_id(0);
    const uint row = get_global_id(1);

    const uint index = row * w + col;

    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint local_index = local_row * get_local_size(0) + local_col;

    const uint group_col = get_group_id(0);
    const uint group_row = get_group_id(1);

    if (col < w && row < h) {
        local_data[local_index] = matrix[index];
    } else {
        local_data[local_index] = 0.0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint transposed_col = group_row * get_local_size(1) + local_col;
    const uint transposed_row = group_col * get_local_size(0) + local_row;

    const uint transposed_index = transposed_row * h + transposed_col;

    const uint transposed_local_index = local_col * get_local_size(1) + local_row;

    if (transposed_col < h && transposed_row < w) {
        transposed_matrix[transposed_index] = local_data[transposed_local_index];
    }
}
