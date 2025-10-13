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
    unsigned int glob_x = get_global_id(0);
    unsigned int glob_y = get_global_id(1);
    unsigned int loc_x = get_local_id(0);
    unsigned int loc_y = get_local_id(1);

    __local float shmem[GROUP_SIZE_X * GROUP_SIZE_Y];
    if (glob_x < w || glob_y < h) {
        shmem[GROUP_SIZE_X * loc_y + (loc_x + loc_y) % GROUP_SIZE_X] = matrix[glob_x + w * glob_y];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (glob_x < w && glob_y < h) {
        transposed_matrix[glob_x * h + glob_y] = shmem[GROUP_SIZE_X * loc_y + (loc_x + loc_y) % GROUP_SIZE_X];
    }
}
