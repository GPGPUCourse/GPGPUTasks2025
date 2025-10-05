#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void
matrix_02_transpose_coalesced_via_local_memory(
    __global const float* matrix, // rows=h x cols=w
    __global float* transposed_matrix, // rows=w x cols=h
    unsigned int w,
    unsigned int h)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int i_local = get_local_id(0);
    const unsigned int j_local = get_local_id(1);

    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];

    if (i >= w || j >= h) {
        local_data[i_local + GROUP_SIZE_X * j_local] = 0;
    } else {
        local_data[i_local + GROUP_SIZE_X * j_local] = matrix[i + w * j];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < w && j < h) {
        transposed_matrix[j + h * i] = local_data[i_local + GROUP_SIZE_X * j_local];
    }
}
