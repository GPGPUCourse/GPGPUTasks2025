#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

// __attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    unsigned int i = get_local_id(0);
    unsigned int j = get_local_id(1);

    unsigned int g_i = get_global_id(0);
    unsigned int g_j = get_global_id(1);

    // printf("liliput #%ld %ld does: %ld %ld\n", i, j, g_i, g_j);

    __local float matrix_cache[GROUP_SIZE];

    unsigned int local_id = j * GROUP_SIZE_X + i;
    unsigned int global_id       = g_j * w + g_i;
    unsigned int global_id_trans = g_j     + g_i * h;


    if (local_id < GROUP_SIZE && global_id < w * h)
        matrix_cache[local_id] = matrix[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < GROUP_SIZE && global_id < w * h)
        transposed_matrix[global_id_trans] = matrix_cache[local_id];
}
