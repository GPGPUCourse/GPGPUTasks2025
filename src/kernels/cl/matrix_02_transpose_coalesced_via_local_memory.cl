#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define TILE_SIZE 32

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);
    __local float local_data[TILE_SIZE][TILE_SIZE+1];

    const uint local_col = get_local_id(0);
    const uint local_row = get_local_id(1);

    const uint global_col = get_group_id(0) * TILE_SIZE + local_col;
    const uint global_row = get_group_id(1) * TILE_SIZE + local_row;


    if (global_col < w && global_row < h){
        local_data[local_row][local_col] = matrix[global_row * w + global_col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint transpose_col = get_group_id(1) * TILE_SIZE + local_col;
    const uint  transpose_row = get_group_id(0) * TILE_SIZE + local_row;

    if (transpose_col < h && transpose_row < w){
        transposed_matrix[transpose_row * h + transpose_col] = local_data[local_col][local_row];
    }

}
