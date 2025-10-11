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
    __local float local_data[256];
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint local_id = get_local_id(0) + 16 * get_local_id(1);
    local_data[local_id] = matrix[x + y * w];
    barrier(CLK_LOCAL_MEM_FENCE);
    uint new_x = get_group_id(0) * 16 + get_local_id(1);
    uint new_y = get_group_id(1) * 16 + get_local_id(0);
    transposed_matrix[new_y + new_x * h] = local_data[(local_id >> 4) + ((local_id & 0xF) << 4)];
}
