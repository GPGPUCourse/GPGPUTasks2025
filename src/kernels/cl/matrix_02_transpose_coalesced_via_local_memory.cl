#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float tile[32][33];

    const unsigned int gx = get_global_id(0);
    const unsigned int gy = get_global_id(1);
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    if (gx < w && gy < h) {
        tile[ly][lx] = matrix[gy * w + gx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int tx = get_group_id(1) * 8 + lx;
    const unsigned int ty = get_group_id(0) * 32 + ly;

    if (tx < h && ty < w) {
        transposed_matrix[ty * h + tx] = tile[lx][ly];
    }
}
