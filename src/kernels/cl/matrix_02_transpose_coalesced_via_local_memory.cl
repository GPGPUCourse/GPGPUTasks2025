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
    const unsigned int global_j_m = get_global_id(0);
    const unsigned int global_i_m = get_global_id(1);
    const unsigned int global_index_m = global_i_m * w + global_j_m;


    const unsigned int local_j = get_local_id(0);
    const unsigned int local_i = get_local_id(1);
    const unsigned int local_index_m = local_i * GROUP_SIZE_X + local_j;
    const unsigned int local_index_mt = local_j * GROUP_SIZE_Y + local_i;

    const unsigned int global_j_mt = get_group_id(1) * GROUP_SIZE_X + local_j;
    const unsigned int global_i_mt = get_group_id(0) * GROUP_SIZE_X + local_i;
    const unsigned int global_index_mt = global_i_mt * h + global_j_mt;

    if (global_index_m < w * h) {
        local_data[local_index_m] = matrix[global_index_m];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (global_index_mt < w * h) {
        transposed_matrix[global_index_mt] = local_data[local_index_mt];
    }
}

