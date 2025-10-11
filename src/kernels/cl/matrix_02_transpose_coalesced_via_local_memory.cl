#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define TILE_X GROUP_SIZE_X
#define BLOCK_ROWS GROUP_SIZE_Y

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const uint local_x = get_local_id(0);
    const uint local_y = get_local_id(1);
    const uint x = get_group_id(0) * TILE_X + local_x;
    const uint y = get_group_id(1) * TILE_X + local_y;

    __local float local_data[TILE_X][TILE_X + 1];
    for (uint i = 0; i < TILE_X; i += BLOCK_ROWS) {
        const uint cur_y = y + i;
        if (x < w && cur_y < h) {
            local_data[local_y + i][local_x] = matrix[cur_y * w + x];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint x_trans = get_group_id(1) * TILE_X + local_x;
    const uint y_trans = get_group_id(0) * TILE_X + local_y;
    for (uint i = 0; i < TILE_X; i += BLOCK_ROWS) {
        const uint cur_y = y_trans + i;
        if (x_trans < h && cur_y < w) {
            transposed_matrix[cur_y * h + x_trans] = local_data[local_x][local_y + i];
        }
    }
}
