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
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint size = GROUP_SIZE_X * GROUP_SIZE_Y;
    __local float localData[size];

    const uint localX = get_local_id(0);
    const uint localY = get_local_id(1);
    const uint localIndex = (localY * GROUP_SIZE_X + localX + 1) % size;

    localData[localIndex] = ((x < w && y < h) ? matrix[y * w + x] : NAN);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < w && y < h) {
        transposed_matrix[x * h + y] = localData[localIndex];
    }
}
