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
    __local float local_data[GROUP_SIZE_X * GROUP_SIZE_Y];

    const unsigned int globalX = get_global_id(0);
    const unsigned int globalY = get_global_id(1);

    const unsigned int localX = get_local_id(0);
    const unsigned int localY = get_local_id(1);

    if (globalX < w && globalY < h) {
        local_data[(localX + localY) % GROUP_SIZE_X + GROUP_SIZE_X * localY] = matrix[globalX + w * globalY];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (globalX < w && globalY < h) {
        transposed_matrix[globalX * h + globalY] = local_data[(localX + localY) % GROUP_SIZE_X + GROUP_SIZE_X * localY];
    }
}