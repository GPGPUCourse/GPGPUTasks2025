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
    unsigned int id_x = get_global_id(0);
    unsigned int id_y = get_global_id(1);
    unsigned int local_id_x = get_local_id(0);
    unsigned int local_id_y = get_local_id(1);
    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];
    local_data[local_id_x + local_id_y * GROUP_SIZE_X] = matrix[id_x + id_y * w];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int group_id_x = get_group_id(0) * GROUP_SIZE_X;
    unsigned int group_id_y = get_group_id(1) * GROUP_SIZE_Y;
    transposed_matrix[group_id_y + group_id_x * h + local_id_x + local_id_y * h] = local_data[local_id_y + local_id_x * GROUP_SIZE_Y];
}
