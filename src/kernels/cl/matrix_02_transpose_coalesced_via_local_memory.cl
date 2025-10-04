#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"

#define LOCAL_WIDTH (GROUP_SIZE_X + 1)
#define LOCAL_HEIGHT GROUP_SIZE_Y

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    __local float local_data[LOCAL_WIDTH * LOCAL_HEIGHT];
    const unsigned int index = y * w + x;
    local_data[local_y * LOCAL_WIDTH + local_x] = index < w * h ? matrix[index] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int new_x = y - local_y + local_x;
    const unsigned int new_y = x - local_x + local_y;
    transposed_matrix[new_y * h + new_x] = local_data[local_x * LOCAL_WIDTH + local_y];
}
