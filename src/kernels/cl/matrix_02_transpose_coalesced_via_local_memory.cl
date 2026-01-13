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
    // __local float tile[1][1 + 1];
    __local float tile[GROUP_SIZE_X][GROUP_SIZE_Y + 1];

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);
    if (x < w && y < h)
        tile[ly][lx] = matrix[y * w + x];
    barrier(CLK_LOCAL_MEM_FENCE);

    // unsigned int trans_x = get_group_id(1) * 1 + lx;
    // unsigned int trans_y = get_group_id(0) * 1 + ly;
    unsigned int trans_x = get_group_id(1) * GROUP_SIZE_X + lx;
    unsigned int trans_y = get_group_id(0) * GROUP_SIZE_Y + ly;
    if (trans_x < h && trans_y < w)
        transposed_matrix[trans_x * w + trans_y] = tile[lx][ly];
}
