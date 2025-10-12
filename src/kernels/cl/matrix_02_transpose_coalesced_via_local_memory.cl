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
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const uint local_index_i = get_local_id(0);
    const uint local_index_j = get_local_id(1);
    const uint local_index = local_index_j * GROUP_SIZE_X + local_index_i;

    __local float local_data[GROUP_SIZE];

    if (i >= w || j >= h) {
        local_data[local_index] = 0;
    }
    else {
        local_data[local_index] = matrix[j * w + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int new_local_ind = local_index_i * GROUP_SIZE_Y + local_index_j;
    if (i < w && j < h) {
        int start_i = i - local_index_i;
        int start_j = j - local_index_j;
        int transposed_index = start_i * h + start_j + local_index_j * h + local_index_i;
        transposed_matrix[transposed_index] = local_data[new_local_ind];
    }
}
