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
    __local float loc[GROUP_SIZE_X][GROUP_SIZE_Y];
    
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < h && y < w) {
        loc[local_x][local_y] = matrix[x * w + y];
        barrier(CLK_LOCAL_MEM_FENCE);
        transposed_matrix[y * h + x] = loc[local_x][local_y];
    }
}
