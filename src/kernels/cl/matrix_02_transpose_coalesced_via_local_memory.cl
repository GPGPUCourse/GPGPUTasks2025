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
    __local float part[16][17]; 
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    if (x >= w || y >= h) {
        part[lx][ly] = 0;
    } else {
        part[lx][ly] = matrix[y * w + x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    transposed_matrix[x * h + y] = part[lx][ly];
}
