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
    __local unsigned int tmp[GROUP_SIZE_Y * GROUP_SIZE_X];

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    tmp[get_local_id(0) + GROUP_SIZE_X * get_local_id(1)] = matrix[x + y * w];
    x -= get_local_id(0);
    x += get_local_id(1);

    y -= get_local_id(1);
    y += get_local_id(0);

    barrier(CLK_LOCAL_MEM_FENCE);

    transposed_matrix[x * h + y] = tmp[get_local_id(1) + GROUP_SIZE_Y * get_local_id(0)];
}
