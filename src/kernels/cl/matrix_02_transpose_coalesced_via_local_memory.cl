#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define BANK_SIZE 32

#define OFFSET (BANK_SIZE / GROUP_SIZE_X)

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];

    local_data[local_y * GROUP_SIZE_X + (local_x + OFFSET * (local_y / OFFSET)) % GROUP_SIZE_X] = matrix[global_y * w + global_x];
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int group_start_x = global_x - local_x;
    const unsigned int group_start_y = global_y - local_y;
    const unsigned int write_x = group_start_y + local_x;
    const unsigned int write_y = group_start_x + local_y;
    transposed_matrix[write_y * h + write_x] = local_data[local_x * GROUP_SIZE_Y + (local_y + OFFSET * (local_x / OFFSET)) % GROUP_SIZE_Y];
}
