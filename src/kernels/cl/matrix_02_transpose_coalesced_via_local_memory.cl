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
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);

    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);

    __local float tile[GROUP_SIZE_X][GROUP_SIZE_Y + 1];

    // read from global memory to local and transpose right away
    tile[lx][ly] = matrix[gy * w + gx];
    barrier(CLK_LOCAL_MEM_FENCE);

    // write to global memory

    // Point at location (gx - lx, gy - ly) -- top-left corner of a tile will go to (gy - ly, gx - lx)
    // If we want to have coalesed write, then we must write from tile from top-left to bottom-right to appropriate locations in B
    // So from top-left corner if we take cell (lx, ly) then it will go to (gy - ly + lx, gx - lx + ly)

    size_t new_gx = gy - ly + lx;
    size_t new_gy = gx - lx + ly;

    transposed_matrix[new_gy * h + new_gx] = tile[ly][lx];
}
