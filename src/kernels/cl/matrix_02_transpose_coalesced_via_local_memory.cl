#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float tile[TILE_SIZE * TILE_SIZE + 1];

    size_t g_col = get_global_id(0);
    size_t g_row = get_global_id(1);
    size_t l_col = get_local_id(0);
    size_t l_row = get_local_id(1);

    size_t block_x = get_group_id(0);
    size_t block_y = get_group_id(1);

    if (g_col < w && g_row < h) {
        tile[l_row * TILE_SIZE + l_col] = matrix[g_row * w + g_col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    size_t out_x = block_y * TILE_SIZE + l_col;
    size_t out_y = block_x * TILE_SIZE + l_row;

    if (out_x < h && out_y < w) {
        transposed_matrix[out_y * h + out_x] = tile[l_col * TILE_SIZE + l_row];
    }
}
