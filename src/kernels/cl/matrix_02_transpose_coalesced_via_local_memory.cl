#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(__global const float* matrix,
                                                             __global       float* transposed_matrix,
                                                             unsigned int w, unsigned int h)
{
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);
    unsigned int gx = get_global_id(0);
    unsigned int gy = get_global_id(1);

    __local float tile[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    if (gx < w && gy < h) {
        tile[ly][lx] = matrix[gy * w + gx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int out_x = get_group_id(1) * GROUP_SIZE_Y + lx;
    unsigned int out_y = get_group_id(0) * GROUP_SIZE_X + ly;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = tile[lx][ly];
    }
}