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
    // TODO
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);
    unsigned int gx = get_group_id(0);
    unsigned int gy = get_group_id(1);

    unsigned int x = gx * GROUP_SIZE + lx;
    unsigned int y = gy * GROUP_SIZE + ly;

    __local float data[GROUP_SIZE_Y][GROUP_SIZE_X + 1];

    if (x < w && y < h)  
        data[ly][lx] = matrix[y * w + x];
    else
        data[ly][lx] = 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int tx = gy * GROUP_SIZE + lx;
    unsigned int ty = gx * GROUP_SIZE + ly;
    if (ty < w && tx < h) {
        transposed_matrix[ty * h + tx] = data[lx][ly];
    }
}
