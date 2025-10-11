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
    // DONE

    __local float tile[GROUP_SIZE_X][GROUP_SIZE_Y + 1];

    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    unsigned int gx = get_group_id(0) * GROUP_SIZE_X + lx;
    unsigned int gy = get_group_id(1) * GROUP_SIZE_Y + ly;

    if (gx < w && gy < h) {
        tile[ly][lx] = matrix[gy * w + gx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    gx = get_group_id(1) * GROUP_SIZE_Y + lx;
    gy = get_group_id(0) * GROUP_SIZE_X + ly;

    if (gx < w && gy < h) {
        transposed_matrix[gy * h + gx] = tile[lx][ly];
    }
}
