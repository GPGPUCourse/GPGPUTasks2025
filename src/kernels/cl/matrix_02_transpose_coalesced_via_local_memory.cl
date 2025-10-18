#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

//works only with square work groups
__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_X, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float data[GROUP_SIZE_X][GROUP_SIZE_X];
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    if (x < w && y < h) {
        data[lx][(lx + ly) % GROUP_SIZE_X] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int new_x = x - lx + ly;
    const unsigned int new_y = y - ly + lx;

    if (new_x < w && new_y < h) {
        transposed_matrix[new_x * h + new_y] = data[ly][(lx + ly) % GROUP_SIZE_X];
    }
}
