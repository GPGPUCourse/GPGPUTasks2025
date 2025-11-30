#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // w x h
    __global float* transposed_matrix, // h x w
    unsigned int w,
    unsigned int h)
{
    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];

    const uint lx = get_local_id(0);
    const uint ly = get_local_id(1);

    const uint read_gx = get_group_id(0) * GROUP_SIZE_X + lx;
    const uint read_gy = get_group_id(1) * GROUP_SIZE_Y + ly;
    local_data[ly * GROUP_SIZE_Y + lx] = matrix[read_gy * w + read_gx];

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint write_gx = get_group_id(1) * GROUP_SIZE_Y + lx;
    const uint write_gy = get_group_id(0) * GROUP_SIZE_X + ly;
    transposed_matrix[write_gy * h + write_gx] = local_data[lx * GROUP_SIZE_Y + ly];
}
