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
    __local float tile[16][17]; // prevent bank conflicts
    int x = get_global_offset(1);
    int y = get_global_offset(0);
    int i = get_local_id(1);
    int j = get_local_id(0);
    if ((x + i) < h && (y + j) < w) {
        tile[i][j] = matrix[(x + i) * w + (y + j)];
    } else {
        tile[i][j] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((y + i) < h && (x + j) < w) {
        transposed_matrix[(y + i) * h + (x + j)] = tile[j][i];
    }
}
