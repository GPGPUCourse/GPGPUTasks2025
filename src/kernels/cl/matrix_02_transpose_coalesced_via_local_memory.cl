#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_X, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    // TODO
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    local float local_matrix[GROUP_SIZE_Y * (GROUP_SIZE_X + 1)];

    if (x >= w || y >= h) {
        return;
    }

    local_matrix[ly * (GROUP_SIZE_X + 1) + lx] = matrix[y * w + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint gx = get_group_id(0);
    const uint gy = get_group_id(1);

    const uint new_x = gx * GROUP_SIZE_Y + ly;
    const uint new_y = gy * GROUP_SIZE_X + lx;

    if (new_x >= h || new_y >= w) {
        return;
    }
    transposed_matrix[new_x * h + new_y] = local_matrix[lx * (GROUP_SIZE_X + 1) + ly];
}
