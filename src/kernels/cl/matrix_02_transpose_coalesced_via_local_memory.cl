#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1))) // supposed: GROUP_SIZE_X == GROUP_SIZE_Y
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const uint i = get_global_id(1);
    const uint j = get_global_id(0);

    const uint local_i = get_local_id(1);
    const uint local_j = get_local_id(0);

    __local float buffer[GROUP_SIZE_Y][GROUP_SIZE_X];

    uint buf_i = local_j;
    uint buf_j = (local_i + local_j) % GROUP_SIZE_X;
    if (i < h && j < w) {
        buffer[buf_i][buf_j] = matrix[i * w + j];
    } else {
        buffer[buf_i][buf_j] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    buf_i = local_i;
    buf_j = (local_i + local_j) % GROUP_SIZE_X;

    uint transposed_i = get_group_id(0) * GROUP_SIZE_X + local_i;
    uint transposed_j = get_group_id(1) * GROUP_SIZE_Y + local_j;
    if (transposed_i < w && transposed_j < h) {
        transposed_matrix[transposed_i * h + transposed_j] = buffer[buf_i][buf_j];
    }
}
