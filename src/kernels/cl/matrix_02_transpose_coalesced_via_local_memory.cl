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
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);
    if (i < w && j < h) {
        // transposed_matrix[i * h + j] = matrix[j * w + i];
        const size_t loc_i = get_local_id(0);
        const size_t loc_j = get_local_id(1);
        // read already transposed:
        local_data[loc_i * GROUP_SIZE_Y + loc_j] = matrix[j * w + i];

        barrier(CLK_LOCAL_MEM_FENCE);
        // (i, j) -> base -> (i - loc_i, j - loc_j) -> T -> (j - loc_j, i - loc_i) ->
        // -> +offset -> (j - loc_j + loc_i, i - loc_i + loc_j)
        transposed_matrix[(i - loc_i + loc_j) * h + (j - loc_j + loc_i)] = local_data[loc_j * GROUP_SIZE_X + loc_i];
    }
}
