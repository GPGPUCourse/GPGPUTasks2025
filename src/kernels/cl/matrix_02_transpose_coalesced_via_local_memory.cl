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
    // just coalesced read + copy in local mem

    __local float local_data[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    uint x = get_global_id(0);
    uint y = get_global_id(1);

    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    // coords of upper-left corner of tile
    uint tile_x = x - lx;
    uint tile_y = y - ly;

    if (x < w && y < h)
        local_data[ly][lx] = matrix[y * w + x];
    else
        local_data[ly][lx] = 255;

    barrier(CLK_LOCAL_MEM_FENCE);

    // swapped (Y, X) -> (X, Y) and now write is coalesced
    // (we suppose that GROUP_SIZE_X == GROUP_SIZE_Y)
    uint new_y = lx + tile_y;
    uint new_x = ly + tile_x;

    // printf("\ttile: (%d, %d); cell: (%d, %d); new_cell: (%d, %d); loc: (%d, %d)\n", tile_x, tile_y, x, y, new_x, new_y, lx, ly);

    if (new_x >= w || new_y >= h)
        return;

    transposed_matrix[new_x * h + new_y] = local_data[lx][ly];
}
