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
    size_t gx = get_group_id(0);
    size_t gy = get_group_id(1);
    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);

    const uint TILE_X = GROUP_SIZE_X;
    const uint TILE_Y = GROUP_SIZE_Y;

    size_t x = gx * TILE_X + lx;
    size_t y = gy * TILE_Y + ly;

    __local float tile[TILE_Y][TILE_X + 1];

    if (x < w && y < h) {
        tile[ly][lx] = matrix[y * w + x];
    } else {
        tile[ly][lx] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t tx = gy * TILE_Y + lx;
    size_t ty = gx * TILE_X + ly;

    if (tx < h && ty < w) {
        transposed_matrix[ty * h + tx] = tile[lx][ly];
    }
}
