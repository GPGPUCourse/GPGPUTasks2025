#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#ifndef GROUP_SIZE_X
  #define GROUP_SIZE_X 16
#endif
#ifndef GROUP_SIZE_Y
  #define GROUP_SIZE_Y 16
#endif

#ifndef TILE_X
  #define TILE_X GROUP_SIZE_X
#endif
#ifndef TILE_Y
  #define TILE_Y GROUP_SIZE_Y
#endif
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // TODO
    const uint gx = get_group_id(0);
    const uint gy = get_group_id(1);
    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    const uint x = gx * TILE_X + lx;
    const uint y = gy * TILE_Y + ly;

    __local float tile[TILE_Y][TILE_X + 1];

    if (x < w && y < h) {
        tile[ly][lx] = matrix[y * w + x];
    } else {
        tile[ly][lx] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint x_out = gy * TILE_Y + lx;
    const uint y_out = gx * TILE_X + ly;

    if (x_out < h && y_out < w) {
        transposed_matrix[y_out * h + x_out] = tile[lx][ly];
    }
}
