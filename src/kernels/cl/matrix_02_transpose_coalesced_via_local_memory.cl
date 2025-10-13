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
        unsigned int group_x = get_group_id(0);
        unsigned int group_y = get_group_id(1);

        unsigned int local_x = get_local_id(0);
        unsigned int local_y = get_local_id(1);

        __local float local_memory[GROUP_SIZE_X][GROUP_SIZE_Y + 1];

        unsigned int global_read_x = group_x * GROUP_SIZE_X + local_x;
        unsigned int global_read_y = group_y * GROUP_SIZE_Y + local_y;

        if (global_read_x < w && global_read_y < h) {
            local_memory[local_y][local_x] = matrix[global_read_y * w + global_read_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int global_write_x = group_y * GROUP_SIZE_X + local_x;
        unsigned int global_write_y = group_x * GROUP_SIZE_Y + local_y;

        if (global_write_x < h && global_write_y < w) {
            transposed_matrix[global_write_y * h + global_write_x] = local_memory[local_x][local_y];
        }
}
