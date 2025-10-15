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
    const unsigned int loc_x = get_local_id(0);
    const unsigned int loc_y = get_local_id(1);
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    if (x >= w || y >= h) {
        tile[loc_y][loc_x] = 0.0f;
    } else {
        tile[loc_y][loc_x] = matrix[y * w + x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int out_x = get_group_id(1) * GROUP_SIZE_Y + loc_x;
    const unsigned int out_y = get_group_id(0) * GROUP_SIZE_X + loc_y;

    if (out_x >= h || out_y >= w) return;

    transposed_matrix[h * out_y + out_x] = tile[loc_x][loc_y];
}
