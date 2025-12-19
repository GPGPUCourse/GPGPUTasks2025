#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    int i_loc = get_local_id(0);
    int j_loc = get_local_id(1);
    __local float local_data[GROUP_SIZE];

    if (i < w && j < h) {
        local_data[j_loc * GROUP_SIZE_X + i_loc] = matrix[j * w + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int out_x = group_y * GROUP_SIZE_Y + i_loc;
    int out_y = group_x * GROUP_SIZE_X + j_loc;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = local_data[i_loc * GROUP_SIZE_X + j_loc];
    }
}
