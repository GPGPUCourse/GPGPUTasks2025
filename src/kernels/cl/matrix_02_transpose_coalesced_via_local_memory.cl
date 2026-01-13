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
    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    size_t l_x = get_local_id(0);
    size_t l_y = get_local_id(1);

    size_t g_x = get_group_id(0);
    size_t g_y = get_group_id(1);

    size_t x = g_x * GROUP_SIZE_X + l_x;
    size_t y = g_y * GROUP_SIZE_Y + l_y;

    if (x < w && y < h) {
        tile[l_y][l_x] = matrix[w * y + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint out_x = g_y * GROUP_SIZE_Y + l_x;
    const uint out_y = g_x * GROUP_SIZE_X + l_y;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = tile[l_x][l_y];
    }

}
