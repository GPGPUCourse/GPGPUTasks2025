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
    __local float data[GROUP_SIZE_Y][GROUP_SIZE_X + 1];
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);

    const unsigned int x = local_x + GROUP_SIZE_X * group_x;
    const unsigned int y = local_y + GROUP_SIZE_Y * group_y;
    if (x >= w || y >= h) {
        data[local_y][local_x] = .0f;
    } else {
        //printf("%d\n", x + y * w);
        data[local_y][local_x] = matrix[x + y * w];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    const unsigned int tr_x = local_x + GROUP_SIZE_Y * group_y;
    const unsigned int tr_y = local_y + GROUP_SIZE_X * group_x;

    if (tr_x >= h || tr_y >= w) {
        return;
    }
    transposed_matrix[tr_x + tr_y * h] = data[local_x][local_y];
}
