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

    uint up_x = get_group_id(0) * GROUP_SIZE_X;
    uint up_y = get_group_id(1) * GROUP_SIZE_Y;

    uint lx = get_local_id(0);
    uint ly = get_local_id(1);

    __local float local_mem[GROUP_SIZE_Y][GROUP_SIZE_X];

    if (up_x + lx < w && up_y + ly < h) {
        local_mem[ly][(lx+ly)%GROUP_SIZE_X] = matrix[(up_y + ly) * w + (up_x + lx)];
    } else {
        local_mem[ly][(lx+ly)%GROUP_SIZE_X] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    lx, ly = ly, lx;

    if (up_x + lx < h && up_y + ly < w) {
        transposed_matrix[(up_x + lx) * h + (up_y + ly)] = local_mem[ly][(lx+ly)%GROUP_SIZE_X];
    } 
}
