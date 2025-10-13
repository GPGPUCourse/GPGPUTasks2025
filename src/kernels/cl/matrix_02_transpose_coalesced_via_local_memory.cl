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
    const int TILE_WIDTH  = GROUP_SIZE_X;
    const int TILE_HEIGHT = GROUP_SIZE_Y;

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    // +1 to avoid bank conflicts
    __local float tile[TILE_HEIGHT][TILE_WIDTH + 1];

    int input_index = gy * w + gx;

    // coalesced read
    if (gx < w && gy < h)
        tile[ly][lx] = matrix[input_index];
    else
        // it's not necessary but seems like good practice
        tile[ly][lx] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    int transposed_x = group_y * TILE_HEIGHT + ly;
    int transposed_y = group_x * TILE_WIDTH  + lx;
    
    if (transposed_x < h && transposed_y < w) {
        transposed_matrix[transposed_y * h + transposed_x] = tile[ly][lx];
    }
}
