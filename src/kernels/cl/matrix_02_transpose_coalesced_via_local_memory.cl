#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // w x h
    __global float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X];

    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int group_x = get_group_id(0) * GROUP_SIZE_X;
    const unsigned int group_y = get_group_id(1) * GROUP_SIZE_Y;

    unsigned int global_x = group_x + local_x;
    unsigned int global_y = group_y + local_y;
    if (global_x < w && global_y < h)
        tile[local_y][local_x] = matrix[global_y * w + global_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned new_w = h;
    const unsigned new_h = w;
    global_x = group_y + local_x;
    global_y = group_x + local_y;
    if (global_x < new_w && global_y < new_h)
        transposed_matrix[global_y * new_w + global_x] = tile[local_x][local_y];
}
