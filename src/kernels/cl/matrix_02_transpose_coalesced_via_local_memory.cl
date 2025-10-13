#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int index_x = get_global_id(0);
    const unsigned int index_y = get_global_id(1);
    const unsigned int local_index_x = get_local_id(0);
    const unsigned int local_index_y = get_local_id(1);
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);
    const unsigned int local_w = get_local_size(0);
    const unsigned int local_h = get_local_size(1);
    __local float local_data[8][33];
    if (index_x < w && index_y < h) {
        local_data[local_index_y][local_index_x] = matrix[index_y * w + index_x];
    } else {
        local_data[local_index_y][local_index_x] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int transposed_x = group_y * local_h + local_index_x;
    unsigned int transposed_y = group_x * local_w + local_index_y;

    if (local_index_x < local_h && local_index_y < local_w) {
        transposed_matrix[transposed_y * h + transposed_x] = local_data[local_index_x][local_index_y];
    }
}
