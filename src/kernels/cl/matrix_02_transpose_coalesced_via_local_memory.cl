#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    __local float tile[16][16];

    const unsigned int localX = get_local_id(0);
    const unsigned int localY = get_local_id(1);

    const unsigned int globalX = get_global_id(0);
    const unsigned int globalY = get_global_id(1);

    if (globalX < w && globalY < h) {
        tile[localY][localX] = matrix[globalY * w + globalX];
    } else {
        tile[localY][localX] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int groupSizeX = get_local_size(0);
    const unsigned int groupSizeY = get_local_size(1);

    const unsigned int transposedLocalX = (localX + localY) % groupSizeY;
    const unsigned int transposedLocalY = localX;

    const unsigned int groupX = get_group_id(0) * groupSizeX;
    const unsigned int groupY = get_group_id(1) * groupSizeY;

    const unsigned int transposedGlobalY = groupX + transposedLocalX;
    const unsigned int transposedGlobalX = groupY + transposedLocalY;


    if (transposedGlobalY < w && transposedGlobalX < h) {
        transposed_matrix[transposedGlobalY * h + transposedGlobalX] = tile[transposedLocalY][transposedLocalX];
    }
}
