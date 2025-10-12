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
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    if (i >= w || j >= h) {
        return;
    }

    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];
    const uint local_index_x = get_local_id(0);
    const uint local_index_y = get_local_id(1);

    //Чтение

    if (i < w && j < h) {
        local_data[local_index_x + local_index_y * GROUP_SIZE_X] = matrix[i + j * w];
    } else {
        local_data[local_index_x + local_index_y * GROUP_SIZE_X] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint work_group_offset_x = get_group_id(0) * GROUP_SIZE_X;
    const uint work_group_offset_y = get_group_id(1) * GROUP_SIZE_Y;

    const uint new_i = work_group_offset_y + local_index_x;
    const uint new_j = work_group_offset_x + local_index_y;

    //Запись

    const uint matrix_index = local_index_y + local_index_x * GROUP_SIZE_X; 

    transposed_matrix[new_i + new_j * h] = local_data[matrix_index];
}
